import time

import numpy as np

import holoclean
import os
import logging

from repair.featurize.loadfeat_featurized_dataset import LoadFeatFeaturizedDataset


class Executor:
    def __init__(self, hc_values, feature_values):
        self.hc_args = hc_values
        self.feature_args = feature_values

    def setup_hc_repair_engine(self, batch_number):
        # Sets up a HoloClean session.
        self.hc = holoclean.HoloClean(
            **self.hc_args
        ).session

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
        self.hc.ds.first_tid = batch_number * self.feature_args['batch_size']

        # Sets the input base path.
        base_path = self.feature_args['feat_dir'] + self.feature_args['dataset_name'] + '_' + self.feature_args['identifier']

        self.hc.repair_engine.feat_dataset = LoadFeatFeaturizedDataset(self.hc.ds, self.hc.env)
        self.hc.repair_engine.feat_dataset.load_feat(base_path, self.feature_args['batch_size'])

    def train(self):
        # Sets up the models.
        self.hc.repair_engine.setup_repair_model()

        # Trains.
        tic = time.clock()
        total_training_cells = 0
        X_train, Y_train, mask_train, train_cid = \
            self.hc.repair_engine.feat_dataset.get_training_data(self.feature_args['labels'])
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

    def infer(self):
        X_pred, mask_pred, infer_idx, Y_truth = self.hc.repair_engine.feat_dataset.get_infer_data(
            self.feature_args['detector_name'])
        Y_pred = {}
        for attr in self.hc.ds.get_active_attributes():
            logging.debug('Inferring %d instances of attribute %s', X_pred[attr].size(0), attr)
            tic_attr = time.clock()
            Y_pred[attr] = self.hc.repair_engine.repair_model[attr].infer_values(X_pred[attr], mask_pred[attr])
            logging.debug('Done. Elapsed time: %.2f', time.clock() - tic_attr)

            grdt = Y_truth[attr].numpy().flatten()
            Y_assign = Y_pred[attr].data.numpy().argmax(axis=1)
            accuracy = 100. * np.mean(Y_assign == grdt)
            logging.debug("%s, acc = %.2f%%", attr, accuracy)

    def run(self):
        total_tuples = 0
        batch_number = 0
        for tuples_to_read in feature_args['tuples_to_read_list']:
            total_tuples += tuples_to_read
            feature_args['batch_size'] = tuples_to_read
            feature_args['identifier'] = '1-' + str(total_tuples)

            self.setup_hc_repair_engine(batch_number)
            self.train()
            self.infer()

            batch_number += 1


if __name__ == "__main__":
    # Default parameters for HoloClean.
    hc_args = {
        'epochs': 2,
        'threads': 1,
        'verbose': True,
        'print_fw': False,
        'timeout': 3 * 60000,
        'incremental': False,
        'infer_mode': 'dk'
    }

    # Default parameters for Executor.
    feature_args = {
        'project_root': os.environ['HOLOCLEANHOME'],
        'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
        'feat_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
        'dataset_name': 'hospital',
        'entity_col': None,
        'numerical_attrs': None,
        'tuples_to_read_list': [250] * 4,
        'active_attributes': ['ProviderNumber', 'HospitalName', 'Address1', 'City', 'State', 'ZipCode', 'CountyName',
                              'PhoneNumber', 'HospitalType', 'HospitalOwner', 'EmergencyService', 'Condition',
                              'MeasureCode', 'MeasureName', 'Score', 'Sample', 'Stateavg'],
        'labels': 'weak',  # ['weak', 'init', 'truth']
        'detector_name': 'AllDetectors',  # ['NullDetector', 'ViolationDetector', 'ErrorLoaderDetector', 'AllDetectors']
    }

    # Runs the default example.
    executor = Executor(hc_args, feature_args)
    executor.run()
