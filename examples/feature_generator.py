import holoclean
import ujson
import importlib
import os
import torch

from dataset import AuxTables


class Executor:
    def __init__(self, hc_values, feature_values):
        self.hc_args = hc_values
        self.feature_args = feature_values

        # Imports modules to dynamically instantiate HoloClean components (detectors and featurizers).
        self.modules = {
            'detect': {detector_file: importlib.import_module('detect.' + detector_file)
                       for detector_file, _, _ in self.hc_args['detectors']},
            'featurize': {featurizer_file: importlib.import_module('repair.featurize.' + featurizer_file)
                          for featurizer_file in self.hc_args['featurizers'].keys()}
        }

    def run_holoclean_featurization(self, csv_fpath):
        # Sets up a HoloClean session.
        self.hc = holoclean.HoloClean(
            **self.hc_args
        ).session

        # Loads existing data and Denial Constraints.
        self.hc.load_data(self.feature_args['dataset_name'],
                          csv_fpath,
                          entity_col=self.feature_args['entity_col'],
                          numerical_attrs=self.feature_args['numerical_attrs'])

        for (_, detector_name, _) in self.hc_args['detectors']:
            if detector_name == 'ViolationDetector':
                self.hc.load_dcs(self.feature_args['dataset_dir'] + self.feature_args['dataset_name'] + '/' +
                                 self.feature_args['dataset_name'] + '_constraints.txt')
                self.hc.ds.set_constraints(self.hc.get_dcs())
                break

        if self.hc_args['detectors'] is not None:
            # Detects erroneous cells.
            detectors = []
            for detector_file, detector_class, detector_has_params in self.hc_args['detectors']:
                if detector_has_params:
                    params = {'fpath': (self.feature_args['dataset_dir'] + self.feature_args['dataset_name'] + '/' +
                                        self.feature_args['dataset_name'] + '_errors.csv')}
                    detectors.append(getattr(self.modules['detect'][detector_file], detector_class)(**params))
                else:
                    if detector_class == 'ViolationDetector':
                        self.hc.load_dcs(self.feature_args['dataset_dir'] + self.feature_args['dataset_name'] + '/' +
                                         self.feature_args['dataset_name'] + '_constraints.txt')
                        self.hc.ds.set_constraints(self.hc.get_dcs())

                    detectors.append(getattr(self.modules['detect'][detector_file], detector_class)())
            self.hc.detect_errors(detectors)

        if self.feature_args['do_quantization']:
            self.hc.quantize_numericals(self.feature_args['num_attr_groups_bins'])

        # Repairs errors based on the defined features.
        self.hc.generate_domain()
        self.hc.run_estimator()

        featurizers = [
            getattr(self.modules['featurize'][featurizer_file], featurizer_class)()
            for featurizer_file, featurizer_class in self.hc_args['featurizers'].items()
        ]
        self.hc.repair_engine.setup_featurized_ds(featurizers)

    def generate_feature_files(self, batches_read):
        # Gets the generated tensors from FeaturizedDataset.
        tensors = self.hc.repair_engine.feat_dataset.tensor
        weak_labels = self.hc.repair_engine.feat_dataset.weak_labels
        is_clean = self.hc.repair_engine.feat_dataset.is_clean
        class_masks = self.hc.repair_engine.feat_dataset.var_class_mask
        tids = self.hc.repair_engine.feat_dataset.tids
        init_idxs = self.hc.repair_engine.feat_dataset.init_idxs

        # Gets the ground truth.
        self.hc.eval_engine.load_data(name=self.feature_args['dataset_name'] + '_clean',
                                      fpath=(self.feature_args['dataset_dir'] + self.feature_args[
                                          'dataset_name'] + '/' +
                                             self.feature_args['dataset_name'] + '_clean.csv'),
                                      tid_col='tid',
                                      attr_col='attribute',
                                      val_col='correct_val')
        clean_df = self.hc.eval_engine.clean_data.df.set_index(['_tid_', '_attribute_'])
        clean_df.sort_index(inplace=True)

        # Index cell_domain table by _tid_ and attribute to lookup domain.
        domain_df = self.hc.ds.aux_table[AuxTables.cell_domain].df[['_tid_', 'attribute', 'domain']]
        domain_df = domain_df.set_index(['_tid_', 'attribute'])
        domain_df.sort_index(inplace=True)

        # Creates tensors for ground truth.
        ground_truth = {}
        for attr, t in tids.items():
            ground_truth[attr] = t.clone()
            for i in range(0, t.size(0)):
                tid = int(t[i])
                if (tid, attr) not in clean_df.index:
                    continue
                g_truth = clean_df.loc[(tid, attr), '_value_']
                domain = domain_df.loc[(tid, attr), 'domain']
                domain_idx = {val: idx for idx, val in enumerate(domain.split('|||'))}
                # -1 means the ground truth is not part of the domain generated for the cell
                ground_truth[attr][i] = domain_idx.get(g_truth, -1)

        # Computes error tensors.
        errors = {}
        errors_df = {}
        for detector in self.hc.detect_engine.detectors:
            errors[detector.name] = {}
            errors_df[detector.name] = detector.detect_noisy_cells().reset_index()
            if not errors_df[detector.name].empty:
                errors_df[detector.name] = errors_df[detector.name].set_index(['_tid_', 'attribute'])
                errors_df[detector.name].sort_index(inplace=True)

            for attr, t in tids.items():
                errors[detector.name][attr] = t.clone().fill_(0)
                if not errors_df[detector.name].empty:
                    for i in range(0, t.size(0)):
                        tid = int(t[i])
                        if (tid, attr) in errors_df[detector.name].index:
                            errors[detector.name][attr][i] = 1

        # Sets the output file names.
        base_path = self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '_' + self.feature_args['identifier']

        batch_size = batches_read[-1]
        # Get tensors for the 'current' batch of the execution.
        # Gets tids from the 'current' batch.
        batch_tids = self.hc.ds.raw_data.df['_tid_'].tail(batch_size).tolist()

        # Gets the tensor entries from the 'current' batch.
        tensors_last = {}
        weak_labels_last = {}
        is_clean_last = {}
        class_masks_last = {}
        tids_last = {}
        init_idxs_last = {}
        ground_truth_last = {}
        errors_last = {key: {} for key in errors.keys()}
        for attr, t in tids.items():
            tids_index = []
            for i in range(0, t.size(0)):
                tid = int(t[i])
                if tid in batch_tids:
                    tids_index.append(i)
            tids_index_tensor = torch.LongTensor(tids_index)
            tensors_last[attr] = tensors[attr].index_select(0, tids_index_tensor)
            weak_labels_last[attr] = weak_labels[attr].index_select(0, tids_index_tensor)
            is_clean_last[attr] = is_clean[attr].index_select(0, tids_index_tensor)
            class_masks_last[attr] = class_masks[attr].index_select(0, tids_index_tensor)
            tids_last[attr] = tids[attr].index_select(0, tids_index_tensor)
            init_idxs_last[attr] = init_idxs[attr].index_select(0, tids_index_tensor)
            ground_truth_last[attr] = ground_truth[attr].index_select(0, tids_index_tensor)
            for detector_name in errors.keys():
                errors_last[detector_name][attr] = errors[detector_name][attr].index_select(0, tids_index_tensor)

        # Wraps tensors in dictionaries.
        feat = {'tensors': tensors, 'errors': errors,
                'labels': {'weak': weak_labels, 'init': init_idxs, 'truth': ground_truth},
                'is_clean': is_clean, 'class_masks': class_masks, 'tids': tids}

        feat_last = {'tensors': tensors_last, 'errors': errors_last,
                     'labels': {'weak': weak_labels_last, 'init': init_idxs_last, 'truth': ground_truth_last},
                     'is_clean': is_clean_last, 'class_masks': class_masks_last, 'tids': tids_last}

        # Dump the files.
        torch.save(feat, base_path + '.feat')
        # The file name includes the standard batch size even if the last batch has less tuples to be easier to find
        # the file in the loadfeat repairer.
        torch.save(feat_last, base_path + '_last' + str(self.feature_args['tuples_to_read_list'][-1]) + '.feat')

        # Dumps the tensors for every batch to a different file if 'current' batch is the last one regarding the full
        # dataset to save the features that correspond to the statistics for the whole dataset.
        if len(batches_read) == len(self.feature_args['tuples_to_read_list']):
            total_tuples = 0
            for batch_size in batches_read:
                total_tuples += batch_size

                # Get tensors for the 'current' batch of the execution.
                # Gets tids from the 'current' batch.
                batch_tids = self.hc.ds.raw_data.df['_tid_'].head(total_tuples).tail(batch_size).tolist()

                # Gets the tensor entries from the 'current' batch.
                tensors_last = {}
                weak_labels_last = {}
                is_clean_last = {}
                class_masks_last = {}
                tids_last = {}
                init_idxs_last = {}
                ground_truth_last = {}
                errors_last = {key: {} for key in errors.keys()}
                for attr, t in tids.items():
                    tids_index = []
                    for i in range(0, t.size(0)):
                        tid = int(t[i])
                        if tid in batch_tids:
                            tids_index.append(i)
                    tids_index_tensor = torch.LongTensor(tids_index)
                    tensors_last[attr] = tensors[attr].index_select(0, tids_index_tensor)
                    weak_labels_last[attr] = weak_labels[attr].index_select(0, tids_index_tensor)
                    is_clean_last[attr] = is_clean[attr].index_select(0, tids_index_tensor)
                    class_masks_last[attr] = class_masks[attr].index_select(0, tids_index_tensor)
                    tids_last[attr] = tids[attr].index_select(0, tids_index_tensor)
                    init_idxs_last[attr] = init_idxs[attr].index_select(0, tids_index_tensor)
                    ground_truth_last[attr] = ground_truth[attr].index_select(0, tids_index_tensor)
                    for detector_name in errors.keys():
                        errors_last[detector_name][attr] = errors[detector_name][attr].index_select(0, tids_index_tensor)

                # Wraps tensors in a dictionary.
                feat_last = {'tensors': tensors_last, 'errors': errors_last,
                             'labels': {'weak': weak_labels_last, 'init': init_idxs_last, 'truth': ground_truth_last},
                             'is_clean': is_clean_last, 'class_masks': class_masks_last, 'tids': tids_last}

                # Dump the file.
                torch.save(feat_last, base_path + '_' + str(total_tuples - batch_size + 1) + '-' +
                           str(total_tuples) + '.feat')

        # Saves stats and environment variables.
        cell_domain = self.hc.ds.aux_table[AuxTables.cell_domain].df.sort_values(by=['_vid_'])
        cell_domain.to_csv(base_path + '_cell_domain.csv', header=True, index=False)

        with open(base_path + '_single_attr_stats.ujson', 'w', encoding='utf-8') as f:
            ujson.dump(self.hc.ds.single_attr_stats, f, ensure_ascii=False)
        with open(base_path + '_pair_attr_stats.ujson', 'w', encoding='utf-8') as f:
            ujson.dump(self.hc.ds.pair_attr_stats, f, ensure_ascii=False)
        with open(base_path + '_num_tuples.txt', 'w', encoding='utf-8') as f:
            f.write(str(self.hc.ds.total_tuples) + '\n')

        with open(base_path + '_hc_env.txt', 'w') as hc_env_file:
            hc_env_file.write(str(self.hc.env))

    def run(self):
        with open(self.feature_args['dataset_dir'] + self.feature_args['dataset_name'] + '/' +
                  self.feature_args['dataset_name'] + '.csv') as dataset_file:

            # Writes the header to a temporary file.
            csv_fpath = '/tmp/current_dataset.csv'
            with open(csv_fpath, 'w+') as tmp_file:
                tmp_file.writelines([dataset_file.readline()])

            batches_read = []
            for tuples_to_read in self.feature_args['tuples_to_read_list']:
                batches_read.append(tuples_to_read)
                # Appends to the temporary file the current batch to be loaded.
                with open(csv_fpath, 'a+') as tmp_file:
                    line_list = []
                    for i in range(tuples_to_read):
                        line = dataset_file.readline()
                        if line == '':
                            # EOF was reached.
                            break
                        line_list.append(line)
                    tmp_file.writelines(line_list)
                    # Sets dynamically the batch size to correctly get the size of the last batch.
                    batches_read[-1] = len(line_list)

                self.feature_args['identifier'] = '1-' + str(sum(batches_read))

                self.run_holoclean_featurization(csv_fpath)
                self.generate_feature_files(batches_read)


if __name__ == "__main__":
    # Default parameters for HoloClean.
    hc_args = {
        'detectors': [
            ('nulldetector', 'NullDetector', False),
            ('violationdetector', 'ViolationDetector', False),
            ('errorloaderdetector', 'ErrorsLoaderDetector', True)
        ],
        'featurizers': {'occurattrfeat': 'OccurAttrFeaturizer'},
        'domain_thresh_1': 0,
        'weak_label_thresh': 0.99,
        'max_domain': 10000,
        'cor_strength': 0.6,
        'nb_cor_strength': 0.8,
        'threads': 1,
        'verbose': True,
        'timeout': 3 * 60000,
        'estimator_type': 'NaiveBayes',
        'incremental': False,
        'infer_mode': 'all',
    }

    # Default parameters for Executor.
    feature_args = {
        'project_root': os.environ['HOLOCLEANHOME'],
        'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
        'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
        'dataset_name': 'hospital',
        'entity_col': None,
        'numerical_attrs': None,
        'do_quantization': False,
        'tuples_to_read_list': [250] * 4,
    }

    # Runs the default example.
    executor = Executor(hc_args, feature_args)
    executor.run()
