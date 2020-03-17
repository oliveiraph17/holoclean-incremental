import numpy as np
import pandas as pd

import holoclean
import os
import sys
import logging


# noinspection PyPep8Naming,PyListCreation
class Executor:
    def __init__(self, hc_values, feature_values):
        self.hc = None
        self.hc_args = hc_values
        self.feature_args = feature_values
        self.training_cells = {}

        self.quality_log_fpath = ''
        if self.hc_args['log_repairing_quality']:
            self.quality_log_fpath += (self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' +
                                       self.feature_args['approach'])

        self.time_log_fpath = ''
        if self.hc_args['log_execution_times']:
            self.time_log_fpath += (self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' +
                                    self.feature_args['approach'] + '_time_log.csv')

        self.weight_log_fpath = ''
        if self.hc_args['log_feature_weights']:
            self.weight_log_fpath += (self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' +
                                      self.feature_args['approach'])

        self.features = 'global' if self.hc_args['global_features'] else 'incremental'
        self.infer_mode = self.hc_args['infer_mode']

    def setup_hc_repair_engine(self, batch_number, batch_size):
        # Sets up a HoloClean session.
        self.hc = holoclean.HoloClean(
            **self.hc_args
        ).session

        # Sets up loggers for the experiments.
        self.hc.setup_experiment_loggers(self.quality_log_fpath + '_quality_log.csv',
                                         self.time_log_fpath,
                                         self.weight_log_fpath + '_weight_log.csv')

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

    def compute_attribute_similarity(self, batch_number, batch_size):
        base_path = self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/features/'
        current_batch_range = '_1-' + str(batch_number * batch_size)
        self.hc.ds.total_tuples, self.hc.ds.single_attr_stats, self.hc.ds.pair_attr_stats = \
            self.hc.ds.load_stats(base_path, current_batch_range)
        correlations = self.hc.ds.compute_norm_cond_entropy_corr()
        return correlations

    def get_weight_file(self, attr, batch_number):
        weight_log_path = self.weight_log_fpath + '_' + attr + '_weight_log_corrected.csv'
        weight_df = pd.read_csv(weight_log_path, sep=';')
        weight_df.drop_duplicates(keep='last', inplace=True)
        weight_df = weight_df.loc[(weight_df['infer_mode'] == self.infer_mode) & \
                                  (weight_df['features'] == self.features) & \
                                  (weight_df['batch_number'] == batch_number)]

        return weight_df

    def get_weight_distance(self, batch_number):
        weight_list = []
        for attr in self.hc.ds.get_active_attributes():
            weight_df = self.get_weight_file(attr, batch_number)
            # Appends to the list discarding the first column (batch_number).
            w1 = weight_df.values.tolist()
            w2 = w1[0][1:len(weight_df.columns) - 2]
            weight_list.append(w2)

        weights = np.array(weight_list)
        # Scale the weights per attribute to the interval [0, 1]. np.spacing is a workaround to avoid division by zero.
        w3 = (np.max(weights, axis=1) + np.spacing(0))
        weights = weights / w3[:, None]

        distances = []
        for weight in weights:
            # Gets a list with the distances between weight and each of the weights
            distance = np.linalg.norm(weights - weight, ord=2, axis=1)
            distances.append(distance)

        return distances


    def run(self):
        batch_number = 1
        number_of_batches = len(self.feature_args['tuples_to_read_list'])

        for batch_size in self.feature_args['tuples_to_read_list']:
            if batch_number > self.feature_args['last_batch']:
                break

            if batch_number >= self.feature_args['first_batch']:
                self.setup_hc_repair_engine(batch_number, batch_size)

                corr = self.compute_attribute_similarity(batch_number, batch_size)
                corr_list = []
                for attr in self.hc.ds.get_active_attributes():
                    # Store the correlations in the order of active attributes to match the weights.
                    corr_list.append([corr[attr][attr2] for attr2 in self.hc.ds.get_active_attributes()])
                corr_array = np.array(corr_list)

                dist_array = self.get_weight_distance(batch_number)
                dist_array = 1 - (dist_array / (np.max(dist_array) + np.spacing(0)))

                attr_ids = [[id] for id in np.arange(dist_array.shape[0])]
                dist_array = np.append(attr_ids, dist_array, axis=1)
                corr_array = np.append(attr_ids, corr_array, axis=1)

                np.savetxt(self.feature_args['log_dir'] + self.feature_args['dataset_name'] + \
                           '_corr.csv', corr_array, delimiter=';')
                np.savetxt(self.feature_args['log_dir'] + self.feature_args['dataset_name'] + \
                           '_dist.csv', dist_array, delimiter=';')

            batch_number += 1


if __name__ == "__main__":
    # Default parameters for HoloClean.
    hc_args = {
        'global_features': False,
        'log_repairing_quality': True,
        'log_execution_times': False,
        'log_feature_weights': True,
        'infer_mode': sys.argv[2],
        'global_features': bool(int(sys.argv[3])),
    }

    # Default parameters for Executor.
    if sys.argv[4] == '_':
        entity_col = None
    else:
        entity_col = '_tid_'
    dataset_name = sys.argv[1]

    if dataset_name == 'hospital' or dataset_name == 'hospital_shuffled':
        active_attributes = ['ProviderNumber', 'HospitalName', 'Address1', 'City', 'State', 'ZipCode', 'CountyName',
                              'PhoneNumber', 'HospitalType', 'HospitalOwner', 'EmergencyService', 'Condition',
                              'MeasureCode', 'MeasureName', 'Score', 'Sample', 'Stateavg']
        tuples_to_read = [10] * 100

    elif dataset_name == 'food5k' or dataset_name == 'food5k_shuffled':
        active_attributes = ['inspectionid', 'dbaname', 'akaname', 'license', 'facilitytype', 'risk',
                              'address', 'city', 'state', 'zip', 'inspectiondate', 'inspectiontype', 'results']
        tuples_to_read = [50] * 100

    elif dataset_name == 'nypd6':
        active_attributes = ['ADDR_PCT_CD', 'BORO_NM', 'CRM_ATPT_CPTD_CD', 'JURISDICTION_CODE', 'JURIS_DESC', 'KY_CD',
                             'LAW_CAT_CD', 'LOC_OF_OCCUR_DESC', 'OFNS_DESC', 'PATROL_BORO', 'PD_CD', 'PD_DESC',
                             'PREM_TYP_DESC']
        tuples_to_read = [324] * 100

    elif dataset_name == 'soccer':
        active_attributes = ['name', 'surname', 'birthyear', 'birthplace', 'position', 'team',
                              'city', 'stadium', 'season', 'manager'],
        tuples_to_read = [2000] * 100
    else:
        raise ValueError('Unknown settings for dataset %s' % dataset_name)

    feature_args = {
        'project_root': os.environ['HOLOCLEANHOME'],
        'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
        'feat_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/' + dataset_name + '/features/',
        'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
        'dataset_name': dataset_name,
        'entity_col': entity_col,
        'numerical_attrs': None,
        'approach': 'monitoring',
        'tuples_to_read_list': tuples_to_read,
        'active_attributes': active_attributes,
        'first_batch': int(sys.argv[5]),
        'last_batch': int(sys.argv[6]),
    }

    # Runs the default example.
    executor = Executor(hc_args, feature_args)
    executor.run()
