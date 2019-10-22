import random
from collections import namedtuple
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import pandas as pd

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
                                         next(iter(tensor.values())).size(2),  # Gets the first tensor in the dict.
                                         featurizer.learnable,
                                         featurizer.init_weight,
                                         featurizer.feature_names())
                                for tensor, featurizer in zip(tensors, featurizers)
                                if tensor is not None]
        # tensor = torch.cat([tens for tens in tensors if tens is not None], 2)
        tensor = {}
        for attr in self.ds.get_active_attributes():
            tensor[attr] = torch.cat([tens[attr] for tens in tensors], 2)
        self.tensor = tensor

        tensor_shapes = [str(attr) + ':' + str(t.shape) for attr, t in self.tensor.items()]
        logging.debug('DONE featurization. Feature tensor size: %s', ', '.join(tensor_shapes))

        if self.env['debug_mode']:
            weights_df = pd.DataFrame(self.tensor.reshape(-1, self.tensor.shape[-1]).numpy())
            weights_df.columns = ["{}::{}".format(f.name, featname) for f in featurizers for featname in f.feature_names()]
            weights_df.insert(0, 'vid', np.floor_divide(np.arange(weights_df.shape[0]), self.tensor.shape[1]) + 1)
            weights_df.insert(1, 'val_idx', np.tile(np.arange(self.tensor.shape[1]), self.tensor.shape[0]))
            weights_df.to_pickle('debug/{}_train_features.pkl'.format(self.ds.id))

        # TODO: remove after we validate it is not needed.
        self.in_features = next(iter(self.tensor.values())).size(2)  # Gets the first tensor in the dict.
        logging.debug("generating weak labels...")
        if not self.env['ignore_previous_training_cells'] or self.ds.is_first_batch():
            self.weak_labels, self.is_clean, self.train_cid, self.weak_label_classes = self.generate_weak_labels()
        else:
            self.weak_labels, self.is_clean, self.train_cid = self.generate_weak_labels_with_reduced_cells()
        logging.debug("DONE generating weak labels.")
        logging.debug("generating mask...")
        self.var_class_mask, self.var_to_domsize = self.generate_var_mask()
        logging.debug("DONE generating mask.")

        if self.env['feature_norm']:
            logging.debug("normalizing features...")
            for attr, t in self.tensor.items():
                # normalize within each cell the features
                self.tensor[attr] = F.normalize(t, p=2, dim=1)
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
        if self.ds.previous_training_df is None:
            query = """
            SELECT t1.attribute, weak_label_idx, fixed, (t2._cid_ IS NULL) AS clean, t1._cid_, t1.weak_label, t1._tid_
            FROM {cell_domain} AS t1 LEFT JOIN {dk_cells} AS t2 ON t1._cid_ = t2._cid_
            ORDER BY _vid_;
            """.format(cell_domain=AuxTables.cell_domain.name,
                       dk_cells=AuxTables.dk_cells.name)
        else:
            query = """
            WITH all_cell_domain AS (SELECT * FROM {cell_domain} UNION SELECT * FROM {cell_domain_previous})
            SELECT t1.attribute, weak_label_idx, fixed, (t2._cid_ IS NULL) AS clean, t1._cid_, t1.weak_label, t1._tid_
            FROM all_cell_domain AS t1 LEFT JOIN {dk_cells} AS t2 ON t1._cid_ = t2._cid_
            ORDER BY _vid_;
            """.format(cell_domain=AuxTables.cell_domain.name,
                       cell_domain_previous=AuxTables.cell_domain_previous.name,
                       dk_cells=AuxTables.dk_cells.name)

        res = self.ds.engine.execute_query(query)
        if len(res) == 0:
            logging.warning("No weak labels available. Reduce pruning threshold.")
        labels = {}
        is_clean = {}
        label_classes = {}
        current_training_cells = []
        # count[attr] will match _vid_ in the featurizers because both are generated ordered by _vid_.
        count = {}
        for attr in self.ds.get_active_attributes():
            count[attr] = 0
            num_instances = self.tensor[attr].size(0)
            labels[attr] = -1 * torch.ones(num_instances, 1).type(torch.LongTensor)
            is_clean[attr] = torch.zeros(num_instances, 1).type(torch.LongTensor)
            label_classes[attr] = {}

        for tuple in tqdm(res):
            attr = tuple[0]
            label = int(tuple[1])
            fixed = int(tuple[2])
            clean = int(tuple[3])
            cid = int(tuple[4])
            label_class = tuple[5]
            tid = tuple[6]
            if label != NULL_REPR and (clean or fixed != CellStatus.NOT_SET.value):
                # Considers only not null and clean or fixed cells as weak labels.
                labels[attr][count[attr]] = label
                is_clean[attr][count[attr]] = clean
                if label_class in label_classes[attr].keys():
                    label_classes[attr][label_class].append((tid, count[attr]))
                else:
                    label_classes[attr][label_class] = [(tid, count[attr])]
            count[attr] += 1
            current_training_cells.append(cid)
        return labels, is_clean, current_training_cells, label_classes

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
        mask = {}
        var_to_domsize = {}
        count = {}
        for attr, t in self.tensor.items():
            mask[attr] = torch.zeros(t.size(0), t.size(1))
            var_to_domsize[attr] = {}
            count[attr] = 0

        if self.ds.previous_training_df is None:
            query = 'SELECT _vid_, domain_size, attribute FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
        else:
            query = """
            WITH all_cell_domain AS (SELECT * FROM {cell_domain} UNION SELECT * FROM {cell_domain_previous})
            SELECT _vid_, domain_size, attribute FROM all_cell_domain ORDER BY _vid_;
            """.format(cell_domain=AuxTables.cell_domain.name,
                       cell_domain_previous=AuxTables.cell_domain_previous.name)

        res = self.ds.engine.execute_query(query)

        for tuple in tqdm(res):
            vid = int(tuple[0])
            max_class = int(tuple[1])
            attr = tuple[2]
            mask[attr][count[attr], max_class:] = -10e6
            var_to_domsize[attr][count[attr]] = max_class
            count[attr] += 1
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
        X_train = {}
        Y_train = {}
        mask_train = {}
        for attr in self.ds.get_active_attributes():
            # train_idx = (self.weak_labels[attr] != -1).nonzero()[:, 0]
            train_idx = self.get_train_idx(attr, 15, 'random')
            X_train[attr] = self.tensor[attr].index_select(0, train_idx)
            Y_train[attr] = self.weak_labels[attr].index_select(0, train_idx)
            mask_train[attr] = self.var_class_mask[attr].index_select(0, train_idx)
        return X_train, Y_train, mask_train, self.train_cid

    def get_infer_data(self):
        """
        Retrieves the samples to be inferred i.e. DK cells.
        """
        infer_idx = {}
        X_infer = {}
        mask_infer = {}
        for attr in self.ds.get_active_attributes():
            if self.env['infer_mode'] == 'dk':
                infer_idx[attr] = (self.is_clean[attr] == 0).nonzero()[:, 0]
            elif self.env['infer_mode'] == 'all':
                infer_idx[attr] = torch.arange(0, self.tensor[attr].size(0))

            X_infer[attr] = self.tensor[attr].index_select(0, infer_idx[attr])
            mask_infer[attr] = self.var_class_mask[attr].index_select(0, infer_idx[attr])
        return X_infer, mask_infer, infer_idx

    def get_train_idx(self, attr, budget_per_class=0, choose_option='random'):
        if budget_per_class > 0:
            # Chooses a number of instances per class
            train_idx = (self.weak_labels[attr] != -1)
            for val, instance_list in self.weak_label_classes[attr].items():
                if len(instance_list) > budget_per_class:
                    logging.debug("%s[%s]=%d", attr, val, len(instance_list))
                    if choose_option == 'random':
                        random.shuffle(instance_list)
                    while len(instance_list) > budget_per_class:
                        _, idx = instance_list.pop()  # Removes to keep the instance list up-to-date
                        train_idx[idx] = 0  # Marks entry to not be used for training
            train_idx = train_idx.nonzero()[:, 0]
        else:
            # Default behavior: use all weak labeled cells
            train_idx = (self.weak_labels[attr] != -1).nonzero()[:, 0]

        return train_idx

    def save_train_data(self):
        rows = []
        for attr in self.ds.get_active_attributes():
            for tid_list in self.weak_label_classes[attr].values():
                for tid, _ in tid_list:
                    rows.append({'_tid_': tid, 'attribute': attr})

        # label_classes[attr][label_class] = [(tid, count[attr])]
        previous_train_df = pd.DataFrame(data=rows)
        self.ds.generate_aux_table(AuxTables.previous_training, previous_train_df, store=True)
