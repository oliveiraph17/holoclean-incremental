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
            current_batch_range = str(starting_tuple) + '_' + str(ending_tuple)
            path = self.base_path + '1-' + str(dataset_size) + '_' + current_batch_range + '.feat'
            logging.info('Loading features from: %s', path)
            self.feat_skipping = torch.load(path)
        else:
            number_of_tensors = (ending_tuple - starting_tuple + 1) / batch_size
            feat_list = []
            current_starting_tuple = starting_tuple

            for i in range(number_of_tensors):
                current_batch_range = str(current_starting_tuple) + '_' + str(current_starting_tuple + batch_size)
                path = self.base_path + '1-' + str(dataset_size) + '_' + current_batch_range + '.feat'
                logging.info('Loading features from: %s', path)
                feat_list.append(torch.load(path))
                current_starting_tuple += batch_size

            if len(feat_list) > 1:
                # Stitches the features into a single variable.
                self.feat_last = feat_list[len(feat_list) - 1]

                tensors_list = [feat['tensors'] for feat in feat_list]
                errors_list = [feat['errors'] for feat in feat_list]
                weak_labels_list = [feat['labels']['weak'] for feat in feat_list]
                init_labels_list = [feat['labels']['init'] for feat in feat_list]
                truth_labels_list = [feat['labels']['truth'] for feat in feat_list]
                is_clean_list = [feat['is_clean'] for feat in feat_list]
                class_masks_list = [feat['class_masks'] for feat in feat_list]
                tids_list = [feat['tids'] for feat in feat_list]

                stitched_tensors = torch.cat(tuple(tensors_list), dim=0)
                stitched_errors = torch.cat(tuple(errors_list), dim=0)
                stitched_weak_labels = torch.cat(tuple(weak_labels_list), dim=0)
                stitched_init_labels = torch.cat(tuple(init_labels_list), dim=0)
                stitched_truth_labels = torch.cat(tuple(truth_labels_list), dim=0)
                stitched_is_clean = torch.cat(tuple(is_clean_list), dim=0)
                stitched_class_masks = torch.cat(tuple(class_masks_list), dim=0)
                stitched_tids = torch.cat(tuple(tids_list), dim=0)

                self.feat = {'tensors': stitched_tensors,
                             'errors': stitched_errors,
                             'labels': {'weak': stitched_weak_labels,
                                        'init': stitched_init_labels,
                                        'truth': stitched_truth_labels},
                             'is_clean': stitched_is_clean,
                             'class_masks': stitched_class_masks,
                             'tids': stitched_tids}
            else:
                self.feat = feat_list[0]
                self.feat_last = feat_list[0]

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
