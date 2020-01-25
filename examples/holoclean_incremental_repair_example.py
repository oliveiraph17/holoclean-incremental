import holoclean
import importlib
import logging
import os
import pandas as pd

from dataset import AuxTables


class Executor:
    def __init__(self, hc_values, inc_values):
        self.hc_args = hc_values
        self.inc_args = inc_values

        self.quality_log_fpath = ''
        if self.hc_args['log_repairing_quality']:
            self.quality_log_fpath += (self.inc_args['log_dir'] + self.inc_args['dataset_name'] + '/' +
                                       self.inc_args['approach'] + '_quality_log.csv')

        self.time_log_fpath = ''
        if self.hc_args['log_execution_times']:
            self.time_log_fpath += (self.inc_args['log_dir'] + self.inc_args['dataset_name'] + '/' +
                                    self.inc_args['approach'] + '_time_log.csv')

        self.weight_log_fpath = ''
        if self.hc_args['log_feature_weights']:
            self.weight_log_fpath += (self.inc_args['log_dir'] + self.inc_args['dataset_name'] + '/' +
                                      self.inc_args['approach'] + '_weight_log.csv')

    def run(self):
        # Imports modules to dynamically instantiate HoloClean components (detectors and featurizers).
        modules = {
            'detect': {detector_file: importlib.import_module('detect.' + detector_file)
                       for detector_file, _, _ in self.hc_args['detectors']},
            'featurize': {featurizer_file: importlib.import_module('repair.featurize.' + featurizer_file)
                          for featurizer_file in self.hc_args['featurizers'].keys()}
        }

        for current_iteration in self.inc_args['iterations']:
            self.hc_args['is_first_batch'] = True

            with open(self.inc_args['dataset_dir'] + self.inc_args['dataset_name'] + '/' +
                      self.inc_args['dataset_name'] + '.csv') as dataset_file:
                self.hc_args['current_iteration'] = current_iteration
                list_element_position = 1

                if self.inc_args['model_monitoring']:
                    self.hc_args['current_batch_number'] = self.hc_args['skip_training_starting_batch'] - 1
                else:
                    self.hc_args['current_batch_number'] = 1

                dataset_file_header = dataset_file.readline()

                for tuples_to_read in self.inc_args['tuples_to_read_list']:
                    # Writes to a temporary file the dataset header plus the current batch to be loaded.
                    with open('/tmp/current_batch.csv', 'w') as tmp_file:
                        line_list = [dataset_file_header]
                        for i in range(tuples_to_read):
                            line = dataset_file.readline()
                            if line == '':
                                # EOF was reached.
                                break
                            line_list.append(line)
                        tmp_file.writelines(line_list)

                    if list_element_position == 2:
                        self.hc_args['is_first_batch'] = False

                        if self.inc_args['model_monitoring']:
                            self.hc_args['skip_training'] = True
                            self.hc_args['train_using_all_batches'] = False

                    # Sets up a HoloClean session.
                    hc = holoclean.HoloClean(
                        **self.hc_args
                    ).session

                    # Drops metatables in the first batch.
                    if self.hc_args['is_first_batch']:
                        table_list = [self.inc_args['dataset_name'] + '_' + self.inc_args['approach'] + '_repaired',
                                      'training_cells',
                                      'repaired_table_copy']

                        hc.ds.engine.drop_tables(table_list)

                    # Sets up loggers for the experiments.
                    hc.setup_experiment_loggers(self.quality_log_fpath, self.time_log_fpath, self.weight_log_fpath)

                    # Loads existing data and Denial Constraints.
                    hc.load_data(self.inc_args['dataset_name'] + '_' + self.inc_args['approach'],
                                 '/tmp/current_batch.csv',
                                 entity_col=self.inc_args['entity_col'],
                                 numerical_attrs=self.inc_args['numerical_attrs'])

                    hc.load_dcs(self.inc_args['dataset_dir'] + self.inc_args['dataset_name'] + '/' +
                                self.inc_args['dataset_name'] + '_constraints.txt')
                    hc.ds.set_constraints(hc.get_dcs())

                    if self.hc_args['infer_mode'] == 'dk':
                        # Detects erroneous cells.
                        detectors = []
                        for detector_file, detector_class, detector_has_params in self.hc_args['detectors']:
                            if detector_has_params:
                                params = {'fpath': (self.inc_args['dataset_dir'] + self.inc_args['dataset_name'] + '/' +
                                                    self.inc_args['dataset_name'] + '_errors.csv')}
                                detectors.append(getattr(modules['detect'][detector_file], detector_class)(**params))
                            else:
                                detectors.append(getattr(modules['detect'][detector_file], detector_class)())
                        hc.detect_errors(detectors)
                    elif self.hc_args['infer_mode'] == 'all':
                        # Skips error detection, creating an empty dk_cells table.
                        empty_dk_cells_df = pd.DataFrame(columns=['_tid_', 'attribute', '_cid_'])
                        empty_dk_cells_df['_tid_'] = empty_dk_cells_df['_tid_'].astype(int)
                        empty_dk_cells_df['attribute'] = empty_dk_cells_df['attribute'].astype(str)
                        empty_dk_cells_df['_cid_'] = empty_dk_cells_df['_cid_'].astype(int)

                        hc.ds.generate_aux_table(AuxTables.dk_cells, empty_dk_cells_df, store=True)

                        if self.hc_args['log_repairing_quality']:
                            hc.repairing_quality_metrics.append(str(self.hc_args['current_batch_number']))
                            hc.repairing_quality_metrics.append(str(0))
                        if self.hc_args['log_execution_times']:
                            hc.execution_times.append(str(0))

                    if self.inc_args['do_quantization']:
                        hc.quantize_numericals(self.inc_args['num_attr_groups_bins'])

                    # Repairs errors based on the defined features.
                    hc.generate_domain()
                    hc.run_estimator()

                    featurizers = [
                        getattr(modules['featurize'][featurizer_file], featurizer_class)()
                        for featurizer_file, featurizer_class in self.hc_args['featurizers'].items()
                    ]
                    hc.repair_errors(featurizers)

                    # Evaluates the correctness of the results.
                    hc.evaluate(fpath=(self.inc_args['dataset_dir'] + self.inc_args['dataset_name'] + '/' +
                                       self.inc_args['dataset_name'] + '_clean.csv'),
                                tid_col='tid',
                                attr_col='attribute',
                                val_col='correct_val')

                    logging.info('Batch %s finished.', self.hc_args['current_batch_number'])

                    self.hc_args['current_batch_number'] += 1
                    list_element_position += 1


if __name__ == "__main__":
    # Default parameters for HoloClean.
    hc_args = {
        # 'detectors': [('nulldetector', 'NullDetector', False),
        #               ('violationdetector', 'ViolationDetector', False)],
        'detectors': [('errorloaderdetector', 'ErrorsLoaderDetector', True)],
        'featurizers': {'occurattrfeat': 'OccurAttrFeaturizer'},
        'domain_thresh_1': 0,
        'domain_thresh_2': 0,
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
        'log_repairing_quality': True,
        'log_execution_times': True,
        'log_feature_weights': True,
        'incremental': True,
        'repair_previous_errors': False,
        'recompute_from_scratch': True,
        'skip_training': False,
        'save_load_checkpoint': False,
        'append': False,
        'infer_mode': 'dk',
        'global_features': False,
        'train_using_all_batches': False,
        'is_first_batch': True,
        'skip_training_starting_batch': -1
    }

    # Default parameters for Executor.
    inc_args = {
        'project_root': os.environ['HOLOCLEANHOME'],
        'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
        'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
        'dataset_name': 'hospital_numerical',
        'entity_col': None,
        'numerical_attrs': ['Score', 'Sample'],
        'do_quantization': True,
        'num_attr_groups_bins': [(100, ['Score']), (150, ['Sample'])],
        'tuples_to_read_list': [1000],
        'model_monitoring': False,
        'dataset_size': None,
        'dataset_fraction_for_batch': None,
        'iterations': [1],
        'approach': 'co_full'
    }

    # Runs the default example.
    executor = Executor(hc_args, inc_args)
    executor.run()
