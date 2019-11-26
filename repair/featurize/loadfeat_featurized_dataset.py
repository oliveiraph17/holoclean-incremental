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


class LoadFeatFeaturizedDataset:
    def __init__(self, dataset, env):
        self.ds = dataset
        self.env = env
        self.classes = {}
        self.featurizer_info = None
        self.feat = None
        self.feat_last = None
        # This list is only used for backward compatibility reasons.
        self.train_cid = []

    def load_feat(self, base_path, batch_size):
        # Sets the input file names.
        logging.info('Loading features from base_path=%s', base_path)

        # Loads the tensors.
        self.feat = torch.load(base_path + '.feat')
        self.feat_last = torch.load(base_path + '_last' + str(batch_size) + '.feat')

        # Sets feat_info to a single featurizer regardless the features were generated using one or several featurizers.
        self.featurizer_info = [FeatInfo('loadfeat',  # name
                                         next(iter(self.feat['tensors'].values())).size(2),  # size
                                         True,  # learnable
                                         1.0,  # init_weight
                                         [])]  # feature_names

        # Sets the number of classes per attribute from the tensor
        for attr in self.ds.get_active_attributes():
            self.classes[attr] = self.feat['tensors'][attr].size(1)

    def get_training_data(self, labels='weak'):
        """
        get_training_data returns X_train, y_train, and mask_train
        where each row of each tensor is a variable/VID and
        y_train are weak labels for each variable i.e. they are
        set as the initial values.

        This assumes that we have a larger proportion of correct initial values
        and only a small amount of incorrect initial values which allow us
        to train to convergence.
        """
        assert labels in ['weak', 'init', 'truth'], \
            "Labels must be 'weak' for estimated weak labels, 'init' for initial values or 'truth' for ground truth."

        X_train = {}
        Y_train = {}
        mask_train = {}

        for attr in self.ds.get_active_attributes():
            # Train using either weak labels or init values
            if labels == 'weak':
                train_idx = (self.feat['weak_labels'][attr] != -1).nonzero()[:, 0]
                Y_train[attr] = self.feat['weak_labels'][attr].index_select(0, train_idx)
            elif labels == 'init':
                train_idx = (self.feat['init_idxs'][attr] != -1).nonzero()[:, 0]
                Y_train[attr] = self.feat['init_idxs'][attr].index_select(0, train_idx)
            else:
                train_idx = (self.feat['ground_truth'][attr] != -1).nonzero()[:, 0]
                Y_train[attr] = self.feat['ground_truth'][attr].index_select(0, train_idx)

            X_train[attr] = self.feat['tensors'][attr].index_select(0, train_idx)
            mask_train[attr] = self.feat['class_masks'][attr].index_select(0, train_idx)

        return X_train, Y_train, mask_train, self.train_cid

    def get_infer_data(self):
        """
        Retrieves the samples to be inferred and the corresponding ground truth.
        """
        assert self.env['infer_mode'] in ['dk', 'all'], \
            "infer_mode must be 'dk' for dirty cells or 'all' for all cells."

        infer_idx = {}
        X_infer = {}
        mask_infer = {}
        Y_ground_truth = {}

        for attr in self.ds.get_active_attributes():
            if self.env['infer_mode'] == 'dk':
                infer_idx[attr] = (self.feat_last['is_clean'][attr] == 0).nonzero()[:, 0]
            else:
                infer_idx[attr] = torch.arange(0, self.feat_last['tensors'][attr].size(0))

            X_infer[attr] = self.feat_last['tensors'][attr].index_select(0, infer_idx[attr])
            mask_infer[attr] = self.feat_last['class_masks'][attr].index_select(0, infer_idx[attr])
            Y_ground_truth[attr] = self.feat_last['ground_truth'][attr].index_select(0, infer_idx[attr])

        return X_infer, mask_infer, infer_idx, Y_ground_truth
