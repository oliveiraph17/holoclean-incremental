import holoclean
import ujson
import importlib
import os
import torch

from dataset import AuxTables


class Executor:
    def __init__(self, hc_values, feature_values):
        self.hc = None
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

    # noinspection PyUnresolvedReferences
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
                                      fpath=(self.feature_args['dataset_dir'] +
                                             self.feature_args['dataset_name'] + '/' +
                                             self.feature_args['dataset_name'] + '_clean.csv'),
                                      tid_col='tid',
                                      attr_col='attribute',
                                      val_col='correct_val')
        clean_df = self.hc.eval_engine.clean_data.df.set_index(['_tid_', '_attribute_'])
        clean_df.sort_index(inplace=True)

        # Indexes cell_domain table by _tid_ and attribute to lookup domain.
        domain_df = self.hc.ds.aux_table[AuxTables.cell_domain].df[['_tid_', 'attribute', 'domain']]
        domain_df = domain_df.set_index(['_tid_', 'attribute'])
        domain_df.sort_index(inplace=True)

        # Creates tensors for ground-truth.
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
                # -1 means the ground-truth is not part of the domain generated for the cell.
                ground_truth[attr][i] = domain_idx.get(g_truth, -1)

        # Computes error tensors.
        errors = {}
        errors_df = {}
        for detector in self.hc.detect_engine.detectors:
            errors[detector.name] = {}
            errors_df[detector.name] = detector.detect_noisy_cells()
            if not errors_df[detector.name].empty:
                errors_df[detector.name].reset_index()
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
        base_path = (self.feature_args['log_dir'] +
                     self.feature_args['dataset_name'] + '_' +
                     self.feature_args['identifier'])

        batch_size = batches_read[-1]

        # Gets tids from the 'current' batch.
        batch_tids = self.hc.ds.raw_data.df['_tid_'].tail(batch_size).tolist()

        # Wraps tensors in a dictionary.
        feat = {'tensors': tensors, 'errors': errors,
                'labels': {'weak': weak_labels, 'init': init_idxs, 'truth': ground_truth},
                'is_clean': is_clean, 'class_masks': class_masks, 'tids': tids}

        # Gets the tensor entries from the 'current' batch.
        feat_last = {key: {} for key in feat.keys()}
        feat_last['labels'] = {label_type: {} for label_type in feat['labels'].keys()}
        feat_last['errors'] = {error_type: {} for error_type in feat['errors'].keys()}

        for attr, t in feat['tids'].items():
            tids_index = []
            for i in range(0, t.size(0)):
                tid = int(t[i])
                if tid in batch_tids:
                    tids_index.append(i)
            tids_index_tensor = torch.LongTensor(tids_index)
            feat_last['tensors'][attr] = feat['tensors'][attr].index_select(0, tids_index_tensor)
            for detector_name in feat['errors'].keys():
                feat_last['errors'][detector_name][attr] = feat['errors'][detector_name][attr]\
                    .index_select(0, tids_index_tensor)
            for label_type in feat['labels'].keys():
                feat_last['labels'][label_type][attr] = feat['labels'][label_type][attr]\
                    .index_select(0, tids_index_tensor)
            feat_last['is_clean'][attr] = feat['is_clean'][attr].index_select(0, tids_index_tensor)
            feat_last['class_masks'][attr] = feat['class_masks'][attr].index_select(0, tids_index_tensor)
            feat_last['tids'][attr] = feat['tids'][attr].index_select(0, tids_index_tensor)

        # Dumps the files.
        torch.save(feat, base_path + '.feat')
        # The file name includes the standard batch size even if the last batch has less tuples to be easier to find
        # the file in the loadfeat repairer.
        torch.save(feat_last, base_path + '_last' + str(self.feature_args['tuples_to_read_list'][-1]) + '.feat')

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

        # Dumps the tensors for every batch to a different file if 'current' batch is the last one regarding the full
        # dataset to save global features that correspond to the statistics for the whole dataset.
        if len(batches_read) == len(self.feature_args['tuples_to_read_list']):
            base_path = self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '_global_'

            # Dumps again the files regarding the last batch but with different names.
            torch.save(feat, base_path + self.feature_args['identifier'] + '.feat')
            # The file name includes the standard batch size even if the last batch has less tuples to be easier to find
            # the file in the loadfeat repairer.
            torch.save(feat_last, base_path + self.feature_args['identifier'] + '_last' +
                       str(self.feature_args['tuples_to_read_list'][-1]) + '.feat')

            # Generates feat and feat_last tensor files for all batches except the last one.
            total_tuples = 0
            for batch_size in batches_read[:len(batches_read)-1]:
                total_tuples += batch_size

                # Gets tids from the 'all' and 'last' batches.
                batch_tids = {'all': self.hc.ds.raw_data.df['_tid_'].head(total_tuples).tolist(),
                              'last': self.hc.ds.raw_data.df['_tid_'].head(total_tuples).tail(batch_size).tolist()}

                for batch_tid in batch_tids.keys():
                    # Reuses feat_last to generate global features.
                    for attr, t in feat['tids'].items():
                        tids_index = []
                        for i in range(0, t.size(0)):
                            tid = int(t[i])
                            if tid in batch_tids[batch_tid]:
                                tids_index.append(i)
                        tids_index_tensor = torch.LongTensor(tids_index)
                        feat_last['tensors'][attr] = feat['tensors'][attr].index_select(0, tids_index_tensor)
                        for detector_name in feat['errors'].keys():
                            feat_last['errors'][detector_name][attr] = feat['errors'][detector_name][attr]\
                                .index_select(0, tids_index_tensor)
                        for label_type in feat['labels'].keys():
                            feat_last['labels'][label_type][attr] = feat['labels'][label_type][attr]\
                                .index_select(0, tids_index_tensor)
                        feat_last['is_clean'][attr] = feat['is_clean'][attr].index_select(0, tids_index_tensor)
                        feat_last['class_masks'][attr] = feat['class_masks'][attr].index_select(0, tids_index_tensor)
                        feat_last['tids'][attr] = feat['tids'][attr].index_select(0, tids_index_tensor)

                    # Dumps the file.
                    if batch_tid == 'all':
                        torch.save(feat_last, base_path + '1-' + str(total_tuples) + '.feat')
                    else:
                        torch.save(feat_last, base_path + '1-' + str(total_tuples) + '_last' +
                                   str(self.feature_args['tuples_to_read_list'][-1]) + '.feat')

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
        'db_port': 5432
    }

    # Default parameters for Executor.
    dataset_name = 'hospital'
    feature_args = {
        'project_root': os.environ['HOLOCLEANHOME'],
        'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
        'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/' + dataset_name + '/features/',
        'dataset_name': dataset_name,
        'entity_col': None,
        'numerical_attrs': None,
        'do_quantization': False,
        'tuples_to_read_list': [10] * 100
    }

    # Runs the default example.
    executor = Executor(hc_args, feature_args)
    executor.run()

    ############################################################

    feature_args['dataset_name'] = 'hospital_shuffled'

    feature_args['log_dir'] = (os.environ['HOLOCLEANHOME'] + '/experimental_results/' +
                               feature_args['dataset_name'] + '/features/')
    feature_args['entity_col'] = '_tid_'

    executor = Executor(hc_args, feature_args)
    executor.run()

    ############################################################

    feature_args['dataset_name'] = 'food5k'

    hc_args['weak_label_thresh'] = 0.6
    hc_args['max_domain'] = 10000
    hc_args['cor_strength'] = 0.2
    hc_args['nb_cor_strength'] = 0.3

    feature_args['log_dir'] = (os.environ['HOLOCLEANHOME'] + '/experimental_results/' +
                               feature_args['dataset_name'] + '/features/')
    feature_args['entity_col'] = None
    feature_args['tuples_to_read_list'] = [50] * 100

    executor = Executor(hc_args, feature_args)
    executor.run()

    ############################################################

    feature_args['dataset_name'] = 'food5k_shuffled'

    feature_args['log_dir'] = (os.environ['HOLOCLEANHOME'] + '/experimental_results/' +
                               feature_args['dataset_name'] + '/features/')
    feature_args['entity_col'] = '_tid_'

    executor = Executor(hc_args, feature_args)
    executor.run()
