import time

import numpy as np

import holoclean
import os
import logging

from repair.featurize.loadfeat_featurized_dataset import LoadFeatFeaturizedDataset


class Executor:
    def __init__(self, hc_values, feature_values):
        self.hc = None
        self.hc_args = hc_values
        self.feature_args = feature_values

        self.quality_log_fpath = ''
        if self.hc_args['log_repairing_quality']:
            self.quality_log_fpath += (self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' +
                                       self.feature_args['approach'] + '_quality_log.csv')

        self.time_log_fpath = ''
        if self.hc_args['log_execution_times']:
            self.time_log_fpath += (self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' +
                                    self.feature_args['approach'] + '_time_log.csv')

        self.weight_log_fpath = ''
        if self.hc_args['log_feature_weights']:
            self.weight_log_fpath += (self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' +
                                      self.feature_args['approach'] + '_weight_log.csv')


    def setup_hc_repair_engine(self, batch_number, batch_size):
        # Sets up a HoloClean session.
        self.hc = holoclean.HoloClean(
            **self.hc_args
        ).session

        # Sets up loggers for the experiments.
        self.hc.setup_experiment_loggers(self.quality_log_fpath, self.time_log_fpath, self.weight_log_fpath)

        # Sets the needed properties for Dataset.
        # Loads a sample from the raw CSV file to do the basic settings.
        with open(feature_args['dataset_dir'] + feature_args['dataset_name'] + '/' +
                  feature_args['dataset_name'] + '.csv') as dataset_file:

            # Generates a non-empty sample file that has the same structure of the data file.
            csv_fpath = '/tmp/current_dataset.csv'
            # Adds the header.
            line_list = [dataset_file.readline()]
            # Adds a sample of the tuples.
            for i in range(100):
                line_list.append(dataset_file.readline())

            with open(csv_fpath, 'w+') as tmp_file:
                tmp_file.writelines(line_list)

        self.hc.ds.load_data(self.feature_args['dataset_name'],
                             csv_fpath,
                             entity_col=self.feature_args['entity_col'],
                             numerical_attrs=self.feature_args['numerical_attrs'],
                             store_to_db=False)

        # Sets manually the active attributes.
        self.hc.ds._active_attributes = self.feature_args['active_attributes']
        # Defines if it is the first batch ('first_tid=0')
        self.hc.ds.first_tid = (batch_number - 1) * batch_size

        # Sets the input base path.
        base_path = (self.feature_args['feat_dir'] + self.feature_args['dataset_name'] + '_')
        if self.hc_args['global_features']:
            base_path += 'global_'

        self.hc.repair_engine.feat_dataset = LoadFeatFeaturizedDataset(self.hc.ds, self.hc.env, base_path)
        self.hc.repair_engine.feat_dataset.load_feat(batch_number, batch_size)

    # noinspection PyPep8Naming
    def train(self):
        # Sets up the models.
        self.hc.repair_engine.setup_repair_model()

        # Trains.
        tic = time.clock()
        total_training_cells = 0
        X_train, Y_train, mask_train, train_cid = \
            self.hc.repair_engine.feat_dataset.get_training_data(self.feature_args['detectors'],
                                                                 self.feature_args['labels'])
        for attr in self.hc.ds.get_active_attributes():
            logging.info('Training model for %s with %d training examples (cells)', attr, X_train[attr].size(0))
            tic_attr = time.clock()

            self.hc.repair_engine.repair_model[attr].fit_model(X_train[attr], Y_train[attr], mask_train[attr],
                                                               self.hc.env['epochs'])

            if self.hc.env['save_load_checkpoint']:
                tic_checkpoint = time.clock()
                self.hc.repair_engine.repair_model[attr].save_checkpoint('/tmp/checkpoint-' + self.hc.ds.raw_data.name
                                                                         + '-' + attr)
                logging.debug("Checkpointing time %.2f.", time.clock() - tic_checkpoint)

            logging.info('Done. Elapsed time: %.2f', time.clock() - tic_attr)
            total_training_cells += X_train[attr].size(0)
        toc = time.clock()

        logging.info('DONE training repair model')
        logging.debug('Time to fit repair model: %.2f secs', toc - tic)
        logging.debug('Number of training elements: %d', total_training_cells)

    # noinspection PyPep8Naming
    def infer(self, skipping=False):
        X_pred, mask_pred, Y_truth = self.hc.repair_engine.feat_dataset.get_infer_data(
            self.feature_args['detectors'],
            skipping)
        Y_pred = {}

        tic_attr = time.clock()
        for attr in self.hc.ds.get_active_attributes():
            logging.debug('Inferring %d instances of attribute %s', X_pred[attr].size(0), attr)
            Y_pred[attr] = self.hc.repair_engine.repair_model[attr].infer_values(X_pred[attr], mask_pred[attr])

            grdt = Y_truth[attr].numpy().flatten()
            Y_assign = Y_pred[attr].data.numpy().argmax(axis=1)
            accuracy = 100. * np.mean(Y_assign == grdt)
            logging.debug("Previous model accuracy: %.2f.", accuracy)

        logging.debug('Done. Elapsed time: %.2f', time.clock() - tic_attr)

        return Y_pred

    def evaluate(self, Y_pred, train_batch_number, skipping_batch_number, skipping=False):
        eval_metrics = self.hc.repair_engine.feat_dataset.get_evaluation_metrics(Y_pred, skipping)

        global_metrics = {name: [] for name in eval_metrics.keys()}
        for attr in self.hc.ds.get_active_attributes():
            for name, value in eval_metrics.items():
                logging.debug('[' + attr + '] ' + name + ' = %2f', value[attr])
                global_metrics[name].append(value[attr])

        sum_metrics = ['total_errors', 'potential_errors', 'total_repairs', 'correct_repairs', 'detected_errors',
                       'total_repairs_grdt_correct', 'total_repairs_grdt_incorrect']

        agg_metrics = {name: 0 for name in eval_metrics.keys()}
        for name in sum_metrics:
            agg_metrics[name] = sum(global_metrics[name])

        if (agg_metrics['total_repairs_grdt_correct'] + agg_metrics['total_repairs_grdt_incorrect']) > 0:
            agg_metrics['precision'] = (agg_metrics['correct_repairs'] /
                                        (agg_metrics['total_repairs_grdt_correct'] +
                                         agg_metrics['total_repairs_grdt_incorrect']))

        if agg_metrics['total_errors'] > 0:
            agg_metrics['recall'] = (agg_metrics['correct_repairs'] / agg_metrics['total_errors'])

        if (agg_metrics['precision'] + agg_metrics['recall']) > 0:
            agg_metrics['f1'] = ((2 * agg_metrics['precision'] * agg_metrics['recall']) /
                                 (agg_metrics['precision'] + agg_metrics['recall']))

        if agg_metrics['detected_errors'] > 0:
            agg_metrics['repairing_recall'] = (agg_metrics['correct_repairs'] / agg_metrics['detected_errors'])

        if (agg_metrics['precision'] + agg_metrics['repairing_recall']) > 0:
            agg_metrics['repairing_f1'] = ((2 * agg_metrics['precision'] * agg_metrics['repairing_recall']) /
                                           (agg_metrics['precision'] + agg_metrics['repairing_recall']))

        for name, agg_value in agg_metrics.items():
            logging.debug('[Aggregated] ' + name + ' = %2f', agg_value)

        agg_values = [self.hc.env['infer_mode']]  # infer_mode
        agg_values.append('global' if self.hc.env['global_features'] else 'incremental')  # features
        agg_values.append('True')  # train_using_all_batches
        agg_values.append(str(train_batch_number + 1))  # skip_training_starting_batch
        agg_values.append(str(skipping_batch_number))  # batch
        agg_values.append(str(agg_metrics['potential_errors']))  # dk_cells
        agg_values.append('0')  # training_cells
        agg_values.append(str(agg_metrics['precision']))
        agg_values.append(str(agg_metrics['recall']))
        agg_values.append(str(agg_metrics['repairing_recall']))
        agg_values.append(str(agg_metrics['f1']))
        agg_values.append(str(agg_metrics['repairing_f1']))
        agg_values.append(str(agg_metrics['detected_errors']))
        agg_values.append(str(agg_metrics['total_errors']))
        agg_values.append(str(agg_metrics['correct_repairs']))
        agg_values.append(str(agg_metrics['total_repairs']))
        agg_values.append(str(agg_metrics['total_repairs_grdt_correct'] + agg_metrics['total_repairs_grdt_incorrect']))
        agg_values.append(str(agg_metrics['total_repairs_grdt_correct']))
        agg_values.append(str(agg_metrics['total_repairs_grdt_incorrect']))
        agg_values.append('0')  # 'rmse'

        self.hc.experiment_quality_logger.info(';'.join(agg_values))


    def run(self):
        total_tuples = 0
        batch_number = 1
        number_of_batches = len(self.feature_args['tuples_to_read_list'])

        for batch_size in self.feature_args['tuples_to_read_list']:
            total_tuples += batch_size

            self.setup_hc_repair_engine(batch_number, batch_size)
            self.train()
            Y_pred = self.infer()
            self.evaluate(Y_pred, batch_number, batch_number)

            number_of_skipping_batches = number_of_batches - batch_number
            for i in range(number_of_skipping_batches):
                self.hc.repair_engine.feat_dataset.load_feat_skipping(batch_number, batch_size, i + 1)
                Y_pred = self.infer(skipping=True)
                self.evaluate(Y_pred, batch_number, batch_number + i + 1, skipping=True)

            batch_number += 1


if __name__ == "__main__":
    # Default parameters for HoloClean.
    hc_args = {
        'epochs': 20,
        'threads': 1,
        'verbose': True,
        'print_fw': False,
        'timeout': 3 * 60000,
        'incremental': False,
        'infer_mode': 'dk',
        'global_features': False,
        'log_repairing_quality': True,
        'log_execution_times': False,
        'log_feature_weights': False,
    }

    # Default parameters for Executor.
    dataset_name = 'hospital'
    feature_args = {
        'project_root': os.environ['HOLOCLEANHOME'],
        'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
        'feat_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/' + dataset_name + '/features/',
        'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
        'dataset_name': dataset_name,
        'entity_col': None,
        'numerical_attrs': None,
        'approach': 'monitoring',
        'tuples_to_read_list': [250] * 4,
        'active_attributes': ['ProviderNumber', 'HospitalName', 'Address1', 'City', 'State', 'ZipCode', 'CountyName',
                              'PhoneNumber', 'HospitalType', 'HospitalOwner', 'EmergencyService', 'Condition',
                              'MeasureCode', 'MeasureName', 'Score', 'Sample', 'Stateavg'],
        'labels': 'weak',  # one of 'weak', 'init' or 'truth'
        'detectors': ['ErrorLoaderDetector']  # ['NullDetector', 'ViolationDetector', 'ErrorLoaderDetector']
    }

    # Runs the default example.
    executor = Executor(hc_args, feature_args)
    executor.run()
