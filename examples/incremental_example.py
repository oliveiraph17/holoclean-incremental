import importlib

import holoclean
import logging
import os

# Parameters for HoloClean
hc_args = {
    'incremental': True,
    'featurizers': {  # Pattern => file_name: class_name
        'occurattrfeat': 'OccurAttrFeaturizer',
        # 'freqfeat': 'FreqFeaturizer',
        # 'constraintfeat': 'ConstraintFeaturizer',
        # 'embeddingfeat': 'EmbeddingFeaturizer',
        # 'initattrfeat': 'InitAttrFeaturizer',
    },
    'detectors': {  # Pattern => file_name: class_name
        'nulldetector': 'NullDetector',
        'violationdetector': 'ViolationDetector',
        # 'errorloaderdetector': 'ErrorsLoaderDetector',
    },
    'log_repairing_quality': True,
    'log_execution_times': False,
    'epochs': 2,
}

# Parameters for the Executor
inc_args = {
    'project_root': os.environ['HOLOCLEANHOME'],
    'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
    'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
    'dataset_name': 'hospital',
    'tuples_to_read_list': [100] * 10,
    'number_of_iterations': 1,
}

class Executor:

    def __init__(self, hc_args, inc_args):
        self.hc_args = hc_args
        self.inc_args = inc_args

        self.log_fpath = ''

        if self.hc_args['log_repairing_quality']:
            self.log_fpath += self.inc_args['log_dir'] + self.inc_args['dataset_name'] + '/quality_log.csv'

        if self.hc_args['log_execution_times']:
            self.log_fpath += self.inc_args['log_dir'] + self.inc_args['dataset_name'] + '/time_log.csv'

    def run(self):
        with open(self.inc_args['dataset_dir'] + self.inc_args['dataset_name'] + '/' +
                  self.inc_args['dataset_name'] + '.csv') as dataset_file:
            # Import modules for dynamically instantiate the HoloClean components (detectors and featurizers)
            modules = {}
            modules['detect'] = {detector_file: importlib.import_module('detect.' + detector_file)
                       for detector_file in self.hc_args['detectors'].keys()}
            modules['featurize'] = {featurizer_file: importlib.import_module('repair.featurize.' + featurizer_file)
                       for featurizer_file in self.hc_args['featurizers'].keys()}

            dataset_file_header = dataset_file.readline()

            for current_iteration in range(self.inc_args['number_of_iterations']):
                current_batch_number = 0

                for tuples_to_read in self.inc_args['tuples_to_read_list']:
                    # Writes to a temporary file the dataset header plus the current batch to be loaded.
                    with open('/tmp/current_batch.csv', 'w') as tmp_file:
                        line_list = [dataset_file_header]
                        for i in range(tuples_to_read):
                            line = dataset_file.readline()
                            if line == '':  # The end of the file was reached
                                break
                            line_list.append(line)
                        tmp_file.writelines(line_list)

                    # Sets up a HoloClean session.
                    hc = holoclean.HoloClean(
                        **self.hc_args
                    ).session

                    # Drops tables and model checkpoint in the first batch.
                    if current_batch_number == 0:
                        table_list = [self.inc_args['dataset_name'] + '_repaired',
                                      'single_attr_stats',
                                      'pair_attr_stats',
                                      'training_cells',
                                      'repaired_table_copy']

                        hc.ds.engine.drop_tables(table_list)

                        if os.path.exists('/tmp/checkpoint.tar'):
                            os.remove('/tmp/checkpoint.tar')

                    # Sets up logger for the experiments.
                    if self.hc_args['log_repairing_quality']:
                        hc.setup_experiment_logger('repairing_quality_logger', self.log_fpath)
                    elif self.hc['log_execution_times']:
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
                    hc.evaluate(fpath=self.inc_args['dataset_dir'] + self.inc_args['dataset_name'] + '/' +
                                      self.inc_args['dataset_name'] + '_clean.csv',
                                tid_col='tid',
                                attr_col='attribute',
                                val_col='correct_val')

                    logging.info('Batch %s finished.', current_batch_number + 1)
                    current_batch_number += 1


# Runs the default example
executor = Executor(hc_args, inc_args)
executor.run()
