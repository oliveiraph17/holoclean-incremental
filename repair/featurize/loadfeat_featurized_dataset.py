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

        # Loads tensors of the last batch within the newly-loaded tensors.
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
                train_idx = ((not_null_labels.long() - (dirty_cells > 0).long()) > 0).nonzero()[:, 0]

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

        eval_metrics = {'dk_cells': {}, 'training_cells': {}, 'precision': {}, 'recall': {}, 'repairing_recall': {},
                        'f1': {}, 'repairing_f1': {}, 'detected_errors': {}, 'total_errors': {}, 'correct_repairs': {},
                        'total_repairs': {}, 'total_repairs_grdt': {}, 'total_repairs_grdt_correct': {},
                        'total_repairs_grdt_incorrect': {}, 'rmse': {}}

        incorrect_repairs = {}

        for attr in self.ds.get_active_attributes():
            # TODO: set correctly the rmse.
            for name in eval_metrics.keys():
                eval_metrics[name][attr] = 0

            # Computes detected errors: counts how many cells were detected by at least one error detector.
            eval_metrics['dk_cells'][attr] = self.infer_idx[attr].size(0)

            # Discards from the metrics cells whose ground-truth is null or it was not provided (closed-world).
            provided_grdt_idx = (feat['labels']['truth'][attr] != -2).nonzero()[:, 0]

            if provided_grdt_idx.nelement() == 0:
                continue

            # Computes total errors: counts how many initial values differ from the ground-truth.
            total_errors = torch.ne(feat['labels']['init'][attr].index_select(0, provided_grdt_idx),
                                    feat['labels']['truth'][attr].index_select(0, provided_grdt_idx))
            eval_metrics['total_errors'][attr] = sum(total_errors).item()

            if eval_metrics['dk_cells'][attr] == 0:
                continue

            # Computes detected errors after inference: counts how many dirty cells were repaired regardless of
            # being correctly repaired or not.
            eval_metrics['detected_errors'][attr] = len(
                set(self.infer_idx[attr].tolist()).intersection((total_errors == 1).nonzero()[:, 0].tolist())
            )

            # Gets the index of the repair with the highest probability for each inferred cell.
            inf_probs, inf_val_id = Y_pred[attr].max(1)

            # Gets the ids for the inferred cells that have ground-truth.
            valid_inf_idx = set(self.infer_idx[attr].tolist()).intersection(provided_grdt_idx.tolist())
            inf_grdt_idx = torch.LongTensor(sorted(valid_inf_idx))

            # Gets a distinct selector for the inf_val_id as its tensors indices are different from feat[label].
            valid_infer_idx = []
            for i in range(self.infer_idx[attr].size(0)):
                if self.infer_idx[attr][i] in inf_grdt_idx:
                    valid_infer_idx.append(i)
            infer_grdt_idx = torch.LongTensor(valid_infer_idx)

            if inf_grdt_idx.nelement() == 0 or infer_grdt_idx.nelement() == 0:
                continue

            # Computes total repairs: counts how many inferred values are different from the corresponding initial
            # ones.
            repaired_cells = torch.ne(inf_val_id.index_select(0, infer_grdt_idx),
                                      feat['labels']['init'][attr].index_select(0, inf_grdt_idx).squeeze())
            eval_metrics['total_repairs'][attr] = sum(repaired_cells).item()

            if eval_metrics['total_repairs'][attr] == 0:
                continue

            # Gets the indices for the repaired cells that have ground-truth.
            repaired_cells_idx = repaired_cells.nonzero()[:, 0]
            rep_grdt_idx = inf_grdt_idx.index_select(0, repaired_cells_idx)

            # Computes correct repairs: counts how many dirty cells were correctly repaired
            # (i.e., init value != inferred value and inferred value == ground-truth).
            eval_metrics['correct_repairs'][attr] = sum(
                torch.eq(inf_val_id.index_select(0, repaired_cells_idx),
                         feat['labels']['truth'][attr].index_select(0, rep_grdt_idx).squeeze())
            ).item()

            # Logs the incorrect repairs (i.e., init value != inferred value and inferred value != ground-truth).
            incorrect_tensor = torch.ne(inf_val_id.index_select(0, repaired_cells_idx),
                                         feat['labels']['truth'][attr].index_select(0, rep_grdt_idx).squeeze())
            incorrect_tensor_idx = incorrect_tensor.nonzero()[:, 0]
            if incorrect_tensor_idx.nelement() > 0:
                incorrect_feat_ix = rep_grdt_idx.index_select(0, incorrect_tensor_idx)
                incorrect_inf_ix = repaired_cells_idx.index_select(0, incorrect_tensor_idx)
                tt = feat['tids'][attr].index_select(0, incorrect_feat_ix).squeeze().tolist()
                pp = inf_val_id.index_select(0, incorrect_inf_ix).squeeze().tolist()
                th = feat['labels']['truth'][attr].index_select(0, incorrect_feat_ix).squeeze().tolist()
                pps = Y_pred[attr].index_select(0, incorrect_inf_ix).tolist()
                if incorrect_tensor_idx.nelement() > 1:
                    incorrect_repairs[attr] = list(zip(tt, pp, th, pps))
                else:
                    # Forces the tuple not to be converted into a number of list elements but be appended as a tuple.
                    fake_list = []
                    fake_list.append((tt, pp, th, pps))
                    incorrect_repairs[attr] = fake_list

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

            eval_metrics['total_repairs_grdt'][attr] = eval_metrics['total_repairs_grdt_correct'][attr] +\
                                                       eval_metrics['total_repairs_grdt_incorrect'][attr]

            if (eval_metrics['total_repairs_grdt'][attr]) > 0:
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

        return eval_metrics, incorrect_repairs
