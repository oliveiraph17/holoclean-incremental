from collections import namedtuple
import logging
import torch

FeatInfo = namedtuple('FeatInfo', ['name', 'size', 'learnable', 'init_weight', 'feature_names'])


class LoadFeatFeaturizedDataset:
    def __init__(self, dataset, env, base_path):
        self.ds = dataset
        self.env = env
        self.classes = {}
        self.base_path = base_path
        self.featurizer_info = None
        self.feat = None
        self.feat_last = None
        self.feat_skipping = None
        # This list is only used for backward compatibility reasons.
        self.train_cid = []

    def load_feat(self, batch_number, batch_size):
        current_batch_range = '1-' + str(batch_number * batch_size)
        path_prefix = self.base_path + current_batch_range

        # Loads tensors from tuples 1 to (`batch_number` * `batch_size`).
        path = path_prefix + '.feat'
        logging.info('Loading features from: %s', path)
        self.feat = torch.load(path)

        # Loads tensors of the last batch encompassed by the newly-loaded tensors.
        path = path_prefix + '_last' + str(batch_size) + '.feat'
        logging.info('Loading features from: %s', path)
        self.feat_last = torch.load(path)

        # Sets feat_info to a single featurizer regardless of whether the features
        # were generated using one or several featurizers.
        self.featurizer_info = [FeatInfo('loadfeat',  # name
                                         next(iter(self.feat['tensors'].values())).size(2),  # size
                                         True,  # learnable
                                         1.0,  # init_weight
                                         [])]  # feature_names

        # Sets the number of classes per attribute from the tensors.
        for attr in self.ds.get_active_attributes():
            self.classes[attr] = self.feat['tensors'][attr].size(1)

    def load_feat_skipping(self, batch_number, batch_size, skipping_batch_number):
        # Loads the tensors for `skipping_batch_number`.
        current_batch_range = '1-' + str((batch_number + skipping_batch_number) * batch_size)
        path_prefix = self.base_path + current_batch_range
        path = path_prefix + '_last' + str(batch_size) + '.feat'
        logging.info('Loading features from: %s', path)
        self.feat_skipping = torch.load(path)

    def load_global_feat(self, dataset_size, batch_size, starting_tuple, ending_tuple, skipping=False):
        # Loads the tensors regarding global features.
        if skipping:
            # As we are skipping training, we only need to memoize `self.feat_skipping`.
            current_batch_range = str(starting_tuple) + '-' + str(ending_tuple)
            path = self.base_path + '1-' + str(dataset_size) + '_' + current_batch_range + '.feat'
            logging.info('Loading features from: %s', path)
            self.feat_skipping = torch.load(path)
        else:
            number_of_tensors = int((ending_tuple - starting_tuple + 1) / batch_size)
            feat_list = []
            current_starting_tuple = starting_tuple

            for i in range(number_of_tensors):
                current_batch_range = str(current_starting_tuple) + '-' + str(current_starting_tuple + batch_size - 1)
                path = self.base_path + '1-' + str(dataset_size) + '_' + current_batch_range + '.feat'
                logging.info('Loading features from: %s', path)
                feat_list.append(torch.load(path))
                current_starting_tuple += batch_size

            if len(feat_list) > 1:
                # Stitches the features into a single variable.
                stitched_feat = {key: {} for key in feat_list[0].keys()}
                stitched_feat['errors'] = {key: {} for key in feat_list[0]['errors'].keys()}
                stitched_feat['labels'] = {key: {} for key in feat_list[0]['labels'].keys()}

                for attr in self.ds.get_active_attributes():
                    stitched_feat['tensors'][attr] = torch.cat(
                        tuple([feat['tensors'][attr] for feat in feat_list])
                    )
                    for error_type in stitched_feat['errors'].keys():
                        stitched_feat['errors'][error_type][attr] = torch.cat(
                            tuple([feat['errors'][error_type][attr] for feat in feat_list])
                        )
                    for label_type in stitched_feat['labels'].keys():
                        stitched_feat['labels'][label_type][attr] = torch.cat(
                            tuple([feat['labels'][label_type][attr] for feat in feat_list])
                        )
                    stitched_feat['is_clean'][attr] = torch.cat(
                        tuple([feat['is_clean'][attr] for feat in feat_list])
                    )
                    stitched_feat['class_masks'][attr] = torch.cat(
                        tuple([feat['class_masks'][attr] for feat in feat_list])
                    )
                    stitched_feat['tids'][attr] = torch.cat(
                        tuple([feat['tids'][attr] for feat in feat_list])
                    )

                self.feat = stitched_feat
                self.feat_last = feat_list[len(feat_list) - 1]
            else:
                self.feat = feat_list[0]
                self.feat_last = feat_list[0]

            # Sets feat_info to a single featurizer regardless of whether the features
            # were generated using one or several featurizers.
            self.featurizer_info = [FeatInfo('loadfeat',  # name
                                             next(iter(self.feat['tensors'].values())).size(2),  # size
                                             True,  # learnable
                                             1.0,  # init_weight
                                             [])]  # feature_names

            # Sets the number of classes per attribute from the tensors.
            for attr in self.ds.get_active_attributes():
                self.classes[attr] = self.feat['tensors'][attr].size(1)

    # noinspection PyPep8Naming
    def get_training_data(self, label_type='weak'):
        """
        get_training_data returns X_train, y_train, and mask_train
        where each row of each tensor is a variable/VID and
        y_train are weak labels for each variable i.e. they are
        set as the initial values.

        This assumes that we have a larger proportion of correct initial values
        and only a small amount of incorrect initial values which allow us
        to train to convergence.
        """
        assert label_type in ['weak', 'init', 'truth'], \
            "Labels must be 'weak' for estimated weak labels, 'init' for initial values or 'truth' for ground truth."

        X_train = {}
        Y_train = {}
        mask_train = {}

        for attr in self.ds.get_active_attributes():
            # Train using label_type
            train_idx = (self.feat['labels'][label_type][attr] != -1).nonzero()[:, 0]
            Y_train[attr] = self.feat['labels'][label_type][attr].index_select(0, train_idx)
            X_train[attr] = self.feat['tensors'][attr].index_select(0, train_idx)
            mask_train[attr] = self.feat['class_masks'][attr].index_select(0, train_idx)

        return X_train, Y_train, mask_train, self.train_cid

    # noinspection PyPep8Naming
    def get_infer_data(self, detectors, skipping=False):
        """
        Retrieves the samples to be inferred and the corresponding ground truth.
        """
        assert self.env['infer_mode'] in ['dk', 'all'], \
            "infer_mode must be 'dk' for dirty cells or 'all' for all cells."

        feat = self.feat_skipping if skipping else self.feat_last

        infer_idx = {}
        X_infer = {}
        mask_infer = {}
        Y_ground_truth = {}

        for attr in self.ds.get_active_attributes():
            if self.env['infer_mode'] == 'dk':
                assert detectors, "No error detector provided for infer_mode='dk'."
                for detector in detectors:
                    # Generates a tensor per attribute with the number of error detections per cell.
                    if attr not in infer_idx:
                        infer_idx[attr] = (feat['errors'][detector][attr] == 1)
                    else:
                        infer_idx[attr] = infer_idx[attr].add(feat['errors'][detector][attr] == 1)
                # Gets the positions which were spot as dirty by at least one error detector.
                infer_idx[attr] = infer_idx[attr].nonzero()[:, 0]
            else:
                infer_idx[attr] = torch.arange(0, feat['tensors'][attr].size(0))

            X_infer[attr] = feat['tensors'][attr].index_select(0, infer_idx[attr])
            mask_infer[attr] = feat['class_masks'][attr].index_select(0, infer_idx[attr])
            Y_ground_truth[attr] = feat['labels']['truth'][attr].index_select(0, infer_idx[attr])

        return X_infer, mask_infer, Y_ground_truth
