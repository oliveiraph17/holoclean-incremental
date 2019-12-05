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

        self.infer_idx = {}

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
                    stitched_feat['fixed'][attr] = torch.cat(
                        tuple([feat['fixed'][attr] for feat in feat_list])
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
    def get_training_data(self, detectors, label_type='weak'):
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
            if detectors:
                if label_type == 'truth':
                    logging.info("No error detection should be provided for training when using 'truth' labels.")
                dirty_cells = None
                for detector in detectors:
                    # Gets the number of times each cell was detected as potentially dirty.
                    if dirty_cells is None:
                        dirty_cells = (self.feat['errors'][detector][attr] == 1)
                    else:
                        dirty_cells = dirty_cells + (self.feat['errors'][detector][attr] == 1)

                # Gets the indices for the cells that have non-null labels and are clean.
                not_null_labels = (self.feat['labels'][label_type][attr] > -1)
                if label_type == 'weak':
                    # Discards the cells whose fixed cell status is NOT_SET (0) and keeps WEAK_LABEL or SINGLE_VALUE.
                    weak_labelled_cells = (self.feat['fixed'][attr] > 0)
                    # Removes from dirty cells those cells that were weak-labelled.
                    dirty_cells = (dirty_cells > 0).long() - weak_labelled_cells.long()
                train_idx = ((not_null_labels.long() - dirty_cells) > 0).nonzero()[:, 0]

            else:
                logging.info('No error detector was provided. Training using all cells as they are all assumed '
                             'clean.')
                # Gets the indices for all cells that have non-null labels.
                train_idx = (self.feat['labels'][label_type][attr] > -1).nonzero()[:, 0]

            # Train using label_type
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

        X_infer = {}
        mask_infer = {}
        self.infer_idx = {}

        Y_ground_truth = {}

        for attr in self.ds.get_active_attributes():
            if self.env['infer_mode'] == 'dk':
                assert detectors, "No error detector provided for infer_mode='dk'."
                dirty_cells = None
                for detector in detectors:
                    # Gets the number of times each cell was detected as potentially dirty.
                    if dirty_cells is None:
                        dirty_cells = (feat['errors'][detector][attr] == 1)
                    else:
                        dirty_cells = dirty_cells + (feat['errors'][detector][attr] == 1)
                # Gets the positions which were spot as dirty by at least one error detector.
                self.infer_idx[attr] = dirty_cells.nonzero()[:, 0]
            else:
                self.infer_idx[attr] = torch.arange(0, feat['tensors'][attr].size(0))

            X_infer[attr] = feat['tensors'][attr].index_select(0, self.infer_idx[attr])
            mask_infer[attr] = feat['class_masks'][attr].index_select(0, self.infer_idx[attr])

            Y_ground_truth[attr] = feat['labels']['truth'][attr].index_select(0, self.infer_idx[attr])

        return X_infer, mask_infer, Y_ground_truth

    def get_evaluation_metrics(self, Y_pred, skipping=False):
        feat = self.feat_skipping if skipping else self.feat_last

        eval_metrics = {'total_errors': {}, 'potential_errors': {}, 'total_repairs': {}, 'correct_repairs': {},
                        'detected_errors': {}, 'total_repairs_grdt_correct': {}, 'total_repairs_grdt_incorrect': {},
                        'precision': {}, 'recall': {}, 'repairing_recall': {}, 'f1': {}, 'repairing_f1': {}, }

        for attr in self.ds.get_active_attributes():
            # Discards from the metrics cells whose ground-truth is null or it was not provided (closed-world).
            provided_grdt_idx = (feat['labels']['truth'][attr] != -2).nonzero()[:, 0]

            # Computes total errors: counts how many initial values differ from the ground-truth.
            total_errors = torch.ne(feat['labels']['init'][attr].index_select(0, provided_grdt_idx),
                                    feat['labels']['truth'][attr].index_select(0, provided_grdt_idx))
            eval_metrics['total_errors'][attr] = sum(total_errors).item()

            # Computes detected errors: counts how many cells were detected by at least one error detector.
            eval_metrics['potential_errors'][attr] = self.infer_idx[attr].size(0)

            # Gets the ids for the inferred cells that have ground-truth.
            inf_grdt_idx = set(self.infer_idx[attr].tolist()).intersection(provided_grdt_idx.tolist())

            # Computes detected errors after inference: counts how many dirty cells were repaired regardless of
            # being correctly repaired or not.
            eval_metrics['detected_errors'][attr] = len(
                set(inf_grdt_idx).intersection((total_errors == 1).nonzero()[:, 0].tolist())
            )

            eval_metrics['total_repairs'][attr] = 0
            eval_metrics['correct_repairs'][attr] = 0
            eval_metrics['total_repairs_grdt_incorrect'][attr] = 0
            eval_metrics['total_repairs_grdt_correct'][attr] = 0
            eval_metrics['precision'][attr] = 0
            eval_metrics['recall'][attr] = 0
            eval_metrics['f1'][attr] = 0
            eval_metrics['repairing_recall'][attr] = 0
            eval_metrics['repairing_f1'][attr] = 0

            if eval_metrics['potential_errors'][attr] > 0:
                # Gets the index of the repair with the highest probability for each inferred cell.
                inf_probs, inf_val_id = Y_pred[attr].max(1)

                # We need distinct selectors for the inf_val_id and feat[init] tensors as the indices are different.
                inf_grdt_idx = torch.LongTensor(sorted(inf_grdt_idx))
                infer_grdt_idx = (self.infer_idx[attr] == inf_grdt_idx).nonzero()[:, 0]

                # Computes total repairs: counts how many inferred values are different from the corresponding initial
                # ones.
                repaired_cells = torch.ne(inf_val_id.index_select(0, infer_grdt_idx),
                                          feat['labels']['init'][attr].index_select(0, inf_grdt_idx).squeeze())
                eval_metrics['total_repairs'][attr] = sum(repaired_cells).item()

                if eval_metrics['total_repairs'][attr] > 0:
                    # Gets the indices for the repaired cells that have ground-truth.
                    repaired_cells_idx = repaired_cells.nonzero()[:, 0]
                    rep_grdt_idx = inf_grdt_idx.index_select(0, repaired_cells_idx)

                    # Computes correct repairs: counts how many dirty cells were correctly repaired
                    # (i.e., init value != inferred value and inferred value == ground-truth).
                    eval_metrics['correct_repairs'][attr] = sum(
                        torch.eq(inf_val_id.index_select(0, repaired_cells_idx),
                                 feat['labels']['truth'][attr].index_select(0, rep_grdt_idx).squeeze())
                    ).item()

                    # Computes repairs on incorrect cells: counts how many repairs were made in dirty cells regardless
                    # of being correct repairs or not (i.e., init value != inferred value and
                    # init_value != ground-truth).
                    eval_metrics['total_repairs_grdt_incorrect'][attr] = sum(
                        torch.ne(feat['labels']['init'][attr].index_select(0, rep_grdt_idx),
                                 feat['labels']['truth'][attr].index_select(0, rep_grdt_idx))
                    ).item()

                    # Computes repairs on correct cells: counts how many repairs (actually messes) were made in clean
                    # cells (i.e., init value != inferred value and init_value == ground-truth).
                    eval_metrics['total_repairs_grdt_correct'][attr] = sum(
                        torch.eq(feat['labels']['init'][attr].index_select(0, rep_grdt_idx),
                                 feat['labels']['truth'][attr].index_select(0, rep_grdt_idx))
                    ).item()

                    if (eval_metrics['total_repairs_grdt_correct'][attr] +
                            eval_metrics['total_repairs_grdt_incorrect'][attr]) > 0:
                        eval_metrics['precision'][attr] = (eval_metrics['correct_repairs'][attr] /
                                                           (eval_metrics['total_repairs_grdt_correct'][attr] +
                                                            eval_metrics['total_repairs_grdt_incorrect'][attr]))

                    if eval_metrics['total_errors'][attr] > 0:
                        eval_metrics['recall'][attr] = (eval_metrics['correct_repairs'][attr] /
                                                        eval_metrics['total_errors'][attr])

                    if (eval_metrics['precision'][attr] + eval_metrics['recall'][attr]) > 0:
                        eval_metrics['f1'][attr] = ((2 * eval_metrics['precision'][attr] *
                                                     eval_metrics['recall'][attr]) /
                                                    (eval_metrics['precision'][attr] +
                                                     eval_metrics['recall'][attr]))

                    if eval_metrics['detected_errors'][attr] > 0:
                        eval_metrics['repairing_recall'][attr] = (eval_metrics['correct_repairs'][attr] /
                                                                  eval_metrics['detected_errors'][attr])

                    if (eval_metrics['precision'][attr] + eval_metrics['repairing_recall'][attr]) > 0:
                        eval_metrics['repairing_f1'][attr] = ((2 * eval_metrics['precision'][attr] *
                                                     eval_metrics['repairing_recall'][attr]) /
                                                    (eval_metrics['precision'][attr] +
                                                     eval_metrics['repairing_recall'][attr]))

        return eval_metrics
