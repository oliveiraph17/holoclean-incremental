from collections import namedtuple
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import AuxTables, CellStatus
from utils import NULL_REPR

FeatInfo = namedtuple('FeatInfo', ['name', 'size', 'learnable', 'init_weight', 'feature_names'])


class FeaturizedDataset:
    def __init__(self, dataset, env, featurizers):
        self.ds = dataset
        self.env = env
        self.total_vars, self.classes = self.ds.get_domain_info()
        for f in featurizers:
            f.setup_featurizer(self.env, self.ds)
        logging.debug('featurizing training data...')
        tensors = [f.create_tensor() for f in featurizers]
        self.featurizer_info = [FeatInfo(featurizer.name,
                                         tensor.size()[2],
                                         featurizer.learnable,
                                         featurizer.init_weight,
                                         featurizer.feature_names())
                                for tensor, featurizer in zip(tensors, featurizers)
                                if tensor is not None]
        tensor = torch.cat([tens for tens in tensors if tens is not None], 2)
        self.tensor = tensor

        logging.debug('DONE featurization. Feature tensor size: %s', self.tensor.shape)

        if self.env['debug_mode']:
            weights_df = pd.DataFrame(self.tensor.reshape(-1, self.tensor.shape[-1]).numpy())
            weights_df.columns = ["{}::{}".format(f.name, featname) for f in featurizers for featname in f.feature_names()]
            weights_df.insert(0, 'vid', np.floor_divide(np.arange(weights_df.shape[0]), self.tensor.shape[1]) + 1)
            weights_df.insert(1, 'val_idx', np.tile(np.arange(self.tensor.shape[1]), self.tensor.shape[0]))
            weights_df.to_pickle('debug/{}_train_features.pkl'.format(self.ds.id))

        # TODO: remove after we validate it is not needed.
        self.in_features = self.tensor.shape[2]
        logging.debug("generating weak labels...")
        if not self.env['ignore_previous_training_cells'] or self.ds.is_first_batch():
            self.weak_labels, self.is_clean, self.train_cid = self.generate_weak_labels()
        else:
            self.weak_labels, self.is_clean, self.train_cid = self.generate_weak_labels_with_reduced_cells()
        logging.debug("DONE generating weak labels.")
        logging.debug("generating mask...")
        self.var_class_mask, self.var_to_domsize = self.generate_var_mask()
        logging.debug("DONE generating mask.")

        if self.env['feature_norm']:
            logging.debug("normalizing features...")
            n_cells, n_classes, n_feats = self.tensor.shape
            # normalize within each cell the features
            self.tensor = F.normalize(self.tensor, p=2, dim=1)
            logging.debug("DONE feature normalization.")

    def generate_weak_labels(self):
        """
        generate_weak_labels returns a tensor where for each VID we have the
        domain index of the initial value.

        :return: Torch.Tensor of size (# of variables) X 1 where tensor[i][0]
            contains the domain index of the initial value for the i-th
            variable/VID.
        """
        # Generate weak labels for clean cells AND cells that have been weak
        # labelled. Do not train on cells with NULL weak labels (i.e.
        # NULL init values that were not weak labelled).
        query = """
        SELECT _vid_, weak_label_idx, fixed, (t2._cid_ IS NULL) AS clean, t1._cid_
        FROM {cell_domain} AS t1 LEFT JOIN {dk_cells} AS t2 ON t1._cid_ = t2._cid_
        WHERE weak_label != '{null_repr}' AND (t2._cid_ is NULL OR t1.fixed != {cell_status});
        """.format(cell_domain=AuxTables.cell_domain.name,
                   dk_cells=AuxTables.dk_cells.name,
                   null_repr=NULL_REPR,
                   cell_status=CellStatus.NOT_SET.value)
        res = self.ds.engine.execute_query(query)
        if len(res) == 0:
            logging.warning("No weak labels available. Reduce pruning threshold.")
        labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        is_clean = torch.zeros(self.total_vars, 1).type(torch.LongTensor)
        current_training_cells = []
        for tuple in tqdm(res):
            vid = int(tuple[0])
            label = int(tuple[1])
            fixed = int(tuple[2])
            clean = int(tuple[3])
            cid = int(tuple[4])
            labels[vid] = label
            is_clean[vid] = clean
            current_training_cells.append(cid)
        return labels, is_clean, current_training_cells

    def generate_weak_labels_with_reduced_cells(self):
        """
        Generates a tensor of weak labels for clean cells and for weak-labelled potentially erroneous cells,
        ignoring cells that were already used for training in previous batches,
        along with a tensor keeping track of clean cells and of weak-labelled potentially erroneous cells,
        the latter including the cells that were ignored in the tensor of weak labels.
        :return: Two torch.Tensors, 'labels' and 'is_clean',
            where 'labels' has size (# of labelled cells ignoring previous training cells, 1)
            with labels[i][0] containing the weak label for the i-th variable/_vid_,
            and 'is_clean' has size (# of labelled cells, 1),
            with is_clean[i][0] containing 1 for a clean j-th variable or 0 for a potentially erroneous j-th variable.
        """
        # Keeps track of clean cells (True) and potentially dirty cells that have been weak-labelled (False).
        # Cells with NULL weak labels (i.e. NULL init values that were not weak-labelled) are ignored.
        is_clean_query = """
            SELECT _vid_, (t2._cid_ IS NULL) AS clean
            FROM {cell_domain} AS t1 LEFT JOIN {dk_cells} AS t2 ON t1._cid_ = t2._cid_
            WHERE weak_label != '{null_repr}' AND (t2._cid_ IS NULL OR t1.fixed != {cell_status});
        """.format(cell_domain=AuxTables.cell_domain.name,
                   dk_cells=AuxTables.dk_cells.name,
                   null_repr=NULL_REPR,
                   cell_status=CellStatus.NOT_SET.value)

        # Executes query.
        is_clean_result = self.ds.engine.execute_query(is_clean_query)

        if len(is_clean_result) == 0:
            raise Exception("No weak labels available. Reduce pruning threshold.")

        # Instantiates tensors.
        labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        is_clean = torch.zeros(self.total_vars, 1).type(torch.LongTensor)

        # Iterates over the result set to update the 'is_clean' tensor.
        for tup in tqdm(is_clean_result):
            vid = int(tup[0])
            clean = int(tup[1])

            is_clean[vid] = clean

        # This query is similar to the previous one, but ignores cells previously used in training.
        # Instead of returning True or False, it returns the weak label of each cell, along with its _cid_.
        weak_label_query = """
            SELECT _vid_, weak_label_idx, t1._cid_
            FROM {cell_domain} AS t1 LEFT JOIN {dk_cells} AS t2 ON t1._cid_ = t2._cid_
            WHERE weak_label != '{null_repr}' AND (t2._cid_ IS NULL OR t1.fixed != {cell_status})
                AND t1._cid_ NOT IN (SELECT _cid_ FROM {training_cells});
        """.format(cell_domain=AuxTables.cell_domain.name,
                   dk_cells=AuxTables.dk_cells.name,
                   null_repr=NULL_REPR,
                   cell_status=CellStatus.NOT_SET.value,
                   training_cells=AuxTables.training_cells.name)

        # Executes query.
        weak_label_result = self.ds.engine.execute_query(weak_label_query)

        # Instantiates list.
        current_training_cells = []

        # Iterates over the result set to update the 'labels' tensor.
        for tup in tqdm(weak_label_result):
            vid = int(tup[0])
            label = int(tup[1])
            cid = int(tup[2])

            labels[vid] = label
            current_training_cells.append(cid)

        return labels, is_clean, current_training_cells

    def generate_var_mask(self):
        """
        generate_var_mask returns a mask tensor where invalid domain indexes
        for a given variable/VID has value -10e6.

        An invalid domain index is possible since domain indexes are expanded
        to the maximum domain size of a given VID: e.g. if a variable A has
        10 unique values and variable B has 6 unique values, then the last
        4 domain indexes (index 6-9) of variable B are invalid.

        :return: Torch.Tensor of size (# of variables) X (max domain)
            where tensor[i][j] = 0 iff the value corresponding to domain index 'j'
            is valid for the i-th VID and tensor[i][j] = -10e6 otherwise.
        """
        var_to_domsize = {}
        query = 'SELECT _vid_, domain_size FROM %s' % AuxTables.cell_domain.name
        res = self.ds.engine.execute_query(query)
        mask = torch.zeros(self.total_vars,self.classes)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            max_class = int(tuple[1])
            mask[vid, max_class:] = -10e6
            var_to_domsize[vid] = max_class
        return mask, var_to_domsize

    def get_tensor(self):
        return self.tensor

    def get_training_data(self):
        """
        get_training_data returns X_train, y_train, and mask_train
        where each row of each tensor is a variable/VID and
        y_train are weak labels for each variable i.e. they are
        set as the initial values.

        This assumes that we have a larger proportion of correct initial values
        and only a small amount of incorrect initial values which allow us
        to train to convergence.
        """
        train_idx = (self.weak_labels != -1).nonzero()[:,0]
        X_train = self.tensor.index_select(0, train_idx)
        Y_train = self.weak_labels.index_select(0, train_idx)
        mask_train = self.var_class_mask.index_select(0, train_idx)
        return X_train, Y_train, mask_train, self.train_cid

    def get_infer_data(self):
        """
        Retrieves the samples to be inferred i.e. DK cells.
        """
        if self.env['infer_mode'] == 'dk':
            infer_idx = (self.is_clean == 0).nonzero()[:, 0]
        elif self.env['infer_mode'] == 'all':
            infer_idx = torch.arange(0, self.tensor.size(0))

        X_infer = self.tensor.index_select(0, infer_idx)
        mask_infer = self.var_class_mask.index_select(0, infer_idx)
        return X_infer, mask_infer, infer_idx
