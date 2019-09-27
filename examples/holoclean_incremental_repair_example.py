import holoclean
import importlib
import logging
import os


class Executor:
    def __init__(self, hc_values, inc_values):
        self.hc_args = hc_values
        self.inc_args = inc_values

        self.log_fpath = ''

        if self.hc_args['log_repairing_quality']:
            self.log_fpath += (self.inc_args['log_dir'] + self.inc_args['dataset_name'] + '/' +
                               self.inc_args['approach'] + '_quality_log.csv')

        if self.hc_args['log_execution_times']:
            self.log_fpath += (self.inc_args['log_dir'] + self.inc_args['dataset_name'] + '/' +
                               self.inc_args['approach'] + '_time_log.csv')

    def run(self):
        with open(self.inc_args['dataset_dir'] + self.inc_args['dataset_name'] + '/' +
                  self.inc_args['dataset_name'] + '.csv') as dataset_file:
            # Imports modules to dynamically instantiate HoloClean components (detectors and featurizers).
            modules = {
                'detect': {detector_file: importlib.import_module('detect.' + detector_file)
                           for detector_file in self.hc_args['detectors'].keys()},
                'featurize': {featurizer_file: importlib.import_module('repair.featurize.' + featurizer_file)
                              for featurizer_file in self.hc_args['featurizers'].keys()}
            }

            dataset_file_header = dataset_file.readline()

            for current_iteration in range(self.inc_args['number_of_iterations']):
                self.hc_args['current_iteration'] = current_iteration
                self.hc_args['current_batch_number'] = 0

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

                    # Sets up a HoloClean session.
                    hc = holoclean.HoloClean(
                        **self.hc_args
                    ).session

                    # Drops metatables in the first batch.
                    if self.hc_args['current_batch_number'] == 0:
                        table_list = [self.inc_args['dataset_name'] + '_repaired',
                                      'single_attr_stats',
                                      'pair_attr_stats',
                                      'training_cells',
                                      'repaired_table_copy']

                        hc.ds.engine.drop_tables(table_list)

                    # Sets up logger for the experiments.
                    if self.hc_args['log_repairing_quality']:
                        hc.setup_experiment_logger('repairing_quality_logger', self.log_fpath)
                    elif self.hc_args['log_execution_times']:
                        hc.setup_experiment_logger('execution_time_logger', self.log_fpath)

                    # Loads existing data and Denial Constraints.
                    hc.load_data(self.inc_args['dataset_name'], '/tmp/current_batch.csv')
                    hc.load_dcs(self.inc_args['dataset_dir'] + self.inc_args['dataset_name'] + '/' +
                                self.inc_args['dataset_name'] + '_constraints.txt')
                    hc.ds.set_constraints(hc.get_dcs())

                    # Detects erroneous cells using these two detectors.
                    detectors = [
                        getattr(modules['detect'][detector_file], detector_class)()
                        for detector_file, detector_class in self.hc_args['detectors'].items()
                    ]
                    hc.detect_errors(detectors)

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

                    logging.info('Batch %s finished.', self.hc_args['current_batch_number'] + 1)
                    self.hc_args['current_batch_number'] += 1


if __name__ == "__main__":
    # Default parameters for HoloClean.
    hc_args = {
        'detectors': {  # Pattern => file_name: class_name
            'nulldetector': 'NullDetector',
            'violationdetector': 'ViolationDetector',
            # 'errorloaderdetector': 'ErrorsLoaderDetector',
        },
        'featurizers': {  # Pattern => file_name: class_name
            'occurattrfeat': 'OccurAttrFeaturizer',
            # 'freqfeat': 'FreqFeaturizer',
            # 'constraintfeat': 'ConstraintFeaturizer',
            # 'embeddingfeat': 'EmbeddingFeaturizer',
            # 'initattrfeat': 'InitAttrFeaturizer',
        },
        'domain_thresh_1': 0,
        'weak_label_thresh': 0.99,
        'max_domain': 10000,
        'cor_strength': 0.6,
        'nb_cor_strength': 0.8,
        'epochs': 10,
        'threads': 1,
        'verbose': True,
        'timeout': 3 * 60000,
        'estimator_type': 'NaiveBayes',
        'current_iteration': None,
        'current_batch_number': None,
        'log_repairing_quality': True,
        'log_execution_times': False,
        'incremental': True,
        'incremental_entropy': False,
        'default_entropy': False,
        'repair_previous_errors': False,
        'recompute_from_scratch': False,
        'skip_training': False,
        'ignore_previous_training_cells': False,
        'save_load_checkpoint': False,
    }

    # Default parameters for Executor.
    inc_args = {
        'project_root': os.environ['HOLOCLEANHOME'],
        'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
        'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
        'dataset_name': 'hospital',
        'approach': 'a',
        'tuples_to_read_list': [100] * 10,
        'number_of_iterations': 1,
    }

    # Runs the default example.
    executor = Executor(hc_args, inc_args)
    executor.run()
