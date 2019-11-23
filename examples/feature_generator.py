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

        # Sets up a HoloClean session.
        self.hc = holoclean.HoloClean(
            **self.hc_args
        ).session

    def run_holoclean_featurization(self):

        # Loads existing data and Denial Constraints.
        self.hc.load_data(self.feature_args['dataset_name'],
                          self.feature_args['dataset_dir'] + self.feature_args['dataset_name'] + '/' +
                          self.feature_args['dataset_name'] + '.csv',
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

    def generate_feature_files(self):
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
                g_truth = clean_df.loc[(tid, attr), '_value_']
                domain = domain_df.loc[(tid, attr), 'domain']
                domain_idx = {val: idx for idx, val in enumerate(domain.split('|||'))}
                # -1 means the ground truth is not part of the domain generated for the cell
                ground_truth[attr][i] = domain_idx.get(g_truth, -1)

        # Get tensors for the 'current' batch of the execution.
        # Gets tids from the 'current' batch.
        batch_tids = self.hc.ds.raw_data.df['_tid_'].tail(self.feature_args['batch_size']).tolist()

        # Gets the tensor entries from the 'current' batch.
        tensors_last = {}
        weak_labels_last = {}
        is_clean_last = {}
        class_masks_last = {}
        tids_last = {}
        init_idxs_last = {}
        ground_truth_last = {}
        for attr, t in tids.items():
            tids_index = []
            for i in range(0, t.size(0)):
                tid = int(t[i])
                if tid in batch_tids:
                    tids_index.append(i)
            tensors_last[attr] = tensors[attr].index_select(0, torch.LongTensor(tids_index))
            weak_labels_last[attr] = weak_labels[attr].index_select(0, torch.LongTensor(tids_index))
            is_clean_last[attr] = is_clean[attr].index_select(0, torch.LongTensor(tids_index))
            class_masks_last[attr] = class_masks[attr].index_select(0, torch.LongTensor(tids_index))
            tids_last[attr] = tids[attr].index_select(0, torch.LongTensor(tids_index))
            init_idxs_last[attr] = init_idxs[attr].index_select(0, torch.LongTensor(tids_index))
            ground_truth_last[attr] = ground_truth[attr].index_select(0, torch.LongTensor(tids_index))

        # Wraps tensors in dictionaries.
        feat = {'tensors': tensors, 'weak_labels': weak_labels, 'is_clean': is_clean,
                'class_masks': class_masks, 'tids': tids, 'init_idxs': init_idxs,
                'ground_truth': ground_truth}

        feat_last = {'tensors': tensors_last, 'weak_labels': weak_labels_last, 'is_clean': is_clean_last,
                     'class_masks': class_masks_last, 'tids': tids_last, 'init_idxs': init_idxs_last,
                     'ground_truth': ground_truth_last}

        # Sets the output file names.
        base_path = self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '_' + self.feature_args['identifier']

        # Dump the files.
        torch.save(feat, base_path + '.feat')
        torch.save(feat_last, base_path + '_last' + str(self.feature_args['batch_size']) + '.feat')

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
        self.run_holoclean_featurization()
        self.generate_feature_files()


if __name__ == "__main__":
    # Default parameters for HoloClean.
    hc_args = {
        'detectors': [
            # ('nulldetector', 'NullDetector', False),
            ('violationdetector', 'ViolationDetector', False),
            # ('errorloaderdetector', 'ErrorsLoaderDetector', True)
        ],
        'featurizers': {'occurattrfeat': 'OccurAttrFeaturizer'},
        'domain_thresh_1': 0,
        'weak_label_thresh': 0.99,
        'max_domain': 10000,
        'cor_strength': 0.6,
        'nb_cor_strength': 0.8,
        'epochs': 20,
        'threads': 1,
        'verbose': True,
        'print_fw': False,
        'timeout': 3 * 60000,
        'estimator_type': 'NaiveBayes',
        'incremental': False,
    }

    # Default parameters for Executor.
    feature_args = {
        'project_root': os.environ['HOLOCLEANHOME'],
        'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
        'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
        'dataset_name': 'hospital_numerical',
        'entity_col': None,
        'numerical_attrs': ['Score', 'Sample'],
        'do_quantization': False,
        'num_attr_groups_bins': None,
        'batch_size': 250,
        'identifier': '1-1000'
    }

    # Runs the default example.
    executor = Executor(hc_args, feature_args)
    executor.run()
