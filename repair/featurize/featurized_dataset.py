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
    # noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
    def __init__(self, dataset, env, featurizers):
        self.ds = dataset
        self.env = env
        self.total_vars, self.classes = self.ds.get_domain_info()
        self.processes = self.env['threads']

        for f in featurizers:
            f.setup_featurizer(self.ds,
                               self.processes,
                               self.env['batch_size'],
                               self.env['repair_previous_errors'])

        logging.debug('Featurizing training data...')

        tensors = [f.create_tensor() for f in featurizers]

        self.featurizer_info = [FeatInfo(featurizer.name,
                                         tensor.size()[2],
                                         featurizer.learnable,
                                         featurizer.init_weight,
                                         featurizer.feature_names())
                                for tensor, featurizer in zip(tensors, featurizers)]

        # The featurizer tensors are concatenated.
        tensor = torch.cat(tensors, 2)
        self.tensor = tensor

        logging.debug('DONE featurization')

        if self.env['debug_mode']:
            weights_df = pd.DataFrame(self.tensor.reshape(-1, self.tensor.shape[-1]).numpy())
            weights_df.columns = ["{}::{}".format(f.name, featname)
                                  for f in featurizers for featname in f.feature_names()]
            weights_df.insert(0, 'vid', np.floor_divide(np.arange(weights_df.shape[0]), self.tensor.shape[1]) + 1)
            weights_df.insert(1, 'val_idx', np.tile(np.arange(self.tensor.shape[1]), self.tensor.shape[0]))
            weights_df.to_pickle('debug/{}_train_features.pkl'.format(self.ds.id))

        # TODO: Remove after we validate it is not needed.
        self.in_features = self.tensor.shape[2]

        logging.debug("Generating weak labels...")
        self.weak_labels, self.is_clean = self.generate_weak_labels()
        logging.debug("DONE generating weak labels")

        logging.debug("Generating mask...")
        self.var_class_mask, self.var_to_domsize = self.generate_var_mask()
        logging.debug("DONE generating mask")

        if self.env['feature_norm']:
            logging.debug("Normalizing features...")
            # n_cells, n_classes, n_feats = self.tensor.shape

            # Normalize the features within each cell.
            self.tensor = F.normalize(self.tensor, p=2, dim=1)
            logging.debug("DONE feature normalization")

    # noinspection PyUnresolvedReferences
    def generate_weak_labels(self):
        """
        Generates a tensor having the domain index of the initial value for each _vid_.

        :return: Torch.Tensor of size "(# of variables) x 1" where tensor[i][0]
                 contains the domain index of the initial value for the i-th variable/_vid_.
        """
        # Generate weak labels for clean cells AND cells that have been weak-labelled.
        # Do not train on cells with NULL weak labels (i.e. NULL init values that have not been weak-labelled).
        query = """
            SELECT _vid_, weak_label_idx, fixed, (t2._cid_ IS NULL) AS clean
            FROM {cell_domain} AS t1 LEFT JOIN {dk_cells} AS t2 ON t1._cid_ = t2._cid_
            WHERE weak_label != '{null_repr}' AND (t2._cid_ IS NULL OR t1.fixed != {cell_status});
        """.format(cell_domain=AuxTables.cell_domain.name,
                   dk_cells=AuxTables.dk_cells.name,
                   null_repr=NULL_REPR,
                   cell_status=CellStatus.NOT_SET.value)

        res = self.ds.engine.execute_query(query)

        if len(res) == 0:
            raise Exception("No weak labels available. Reduce pruning threshold.")

        labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        is_clean = torch.zeros(self.total_vars, 1).type(torch.LongTensor)

        for tup in tqdm(res):
            vid = int(tup[0])
            label = int(tup[1])
            # The variable 'fixed' is not being used, so the statement has been commented out.
            # fixed = int(tup[2])
            clean = int(tup[3])
            labels[vid] = label
            is_clean[vid] = clean

        return labels, is_clean

    def generate_var_mask(self):
        """
        Returns a mask tensor where invalid domain indexes for a given variable/VID has value -10e6.

        An invalid domain index is possible since domain indexes are expanded to the maximum domain size
        across all _vid_'s. For instance, if variable A has 10 unique values and variable B has 6 unique values,
        then the last 4 domain indexes (index 6-9) of variable B are invalid.

        :return: Torch.Tensor of size "(# of variables) X (maximum domain size)"
                 where tensor[i][j] = 0 iff the value corresponding to domain index 'j' is valid for the i-th _vid_
                 and tensor[i][j] = -10e6 otherwise.
        """
        var_to_domsize = {}
        query = 'SELECT _vid_, domain_size FROM %s' % AuxTables.cell_domain.name
        res = self.ds.engine.execute_query(query)
        mask = torch.zeros(self.total_vars, self.classes)
        for tup in tqdm(res):
            vid = int(tup[0])
            max_class = int(tup[1])
            mask[vid, max_class:] = -10e6
            var_to_domsize[vid] = max_class
        return mask, var_to_domsize

    def get_tensor(self):
        return self.tensor

    def get_training_data(self):
        """
        Returns the tensors 'x_train', 'y_train', and 'mask_train', where each row of each tensor is a variable/_vid_.
        The training data are selected by indexes representing weak labels different than -1 and 0.
        The tensor 'y_train' has weak labels for each variable, i.e. they are set as the initial values.

        This assumes that we have a larger proportion of correct initial values and only
        a small amount of incorrect initial values, which allows us to train for convergence.
        """
        train_idx = (self.weak_labels != -1).nonzero()[:, 0]
        x_train = self.tensor.index_select(0, train_idx)
        y_train = self.weak_labels.index_select(0, train_idx)
        mask_train = self.var_class_mask.index_select(0, train_idx)
        return x_train, y_train, mask_train

    def get_infer_data(self):
        """
        Retrieves the samples to be inferred, i.e. DK cells.
        """
        # Only infer on those that are DK cells.
        infer_idx = (self.is_clean == 0).nonzero()[:, 0]
        x_infer = self.tensor.index_select(0, infer_idx)
        mask_infer = self.var_class_mask.index_select(0, infer_idx)
        return x_infer, mask_infer, infer_idx
