import numpy as np
import pandas as pd

import holoclean
import os
import sys
import logging
import copy

from scipy.stats import entropy

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

    def load_stats(self, batch_number, batch_size):
        base_path = self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/features/'
        current_batch_range = '_1-' + str(batch_number * batch_size)
        return self.hc.ds.load_stats(base_path, current_batch_range)

    def compute_attribute_similarity(self, batch_number, batch_size):
        self.hc.ds.total_tuples, self.hc.ds.single_attr_stats, self.hc.ds.pair_attr_stats = self.load_stats(
            batch_number, batch_size)
        correlations = self.hc.ds.compute_norm_cond_entropy_corr()
        return correlations

    def correct_weight_file(self, attribute):
        weight_log_path = self.weight_log_fpath + '_' + attribute + '_weight_log'
        quality_log_path = self.weight_log_fpath + '_' + attribute + '_quality_log'

        weight_df = pd.read_csv(weight_log_path + '.csv', sep=';')
        quality_df = pd.read_csv(quality_log_path + '.csv', sep=';')

        # Clear extra headers from the middle of the file.
        weight_df = weight_df.loc[weight_df['w0'] != 'w0']
        # Clear log lines not associated to training (but inferring).
        quality_df = quality_df.loc[quality_df['train_batch'] == quality_df['infer_batch']]

        # Align the indexes because when trying to assign a column of one DataFrame to another,
        # pandas will try to align the indexes, and failing to do so, insert NaNs.
        weight_df.index = quality_df.index

        # Assign the columns. It works because the original row orders from the files correspond.
        weight_df['infer_mode'] = quality_df['infer_mode']
        weight_df['features'] = quality_df['features']

        weight_df = weight_df.astype({'batch_number': 'int32'})
        weight_df = weight_df.sort_values(by=['batch_number'])

        weight_df.to_csv(weight_log_path + '_corrected.csv', index=False, sep=';')

    def get_weight_file(self, attr, batch_number):
        weight_log_path = self.weight_log_fpath + '_' + attr + '_weight_log_corrected.csv'
        if not os.path.isfile(weight_log_path):
            self.correct_weight_file(attr)

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


    def generate_attr_corr_and_weight_dist(self):
        batch_number = self.feature_args['first_batch']

        for batch_size in self.feature_args['tuples_to_read_list']:
            if batch_number > self.feature_args['last_batch']:
                break

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

            np.savetxt(self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' + \
                       self.feature_args['dataset_name'] + '_batch' + str(batch_number) + '_corr.csv',
                       corr_array, delimiter=';')
            np.savetxt(self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' + \
                       self.feature_args['dataset_name'] + '_batch' + str(batch_number) + '_dist.csv',
                       dist_array, delimiter=';')

            batch_number += 1

    def get_attribute_groups(self):
        batch_number = self.feature_args['first_batch']

        for batch_size in self.feature_args['tuples_to_read_list']:
            if batch_number > self.feature_args['last_batch']:
                break

            data_corr = np.loadtxt(self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' + \
                       self.feature_args['dataset_name'] + '_batch' + str(batch_number) + '_corr.csv', delimiter=';')
            data_dist = np.loadtxt(self.feature_args['log_dir'] + self.feature_args['dataset_name'] + '/' + \
                       self.feature_args['dataset_name'] + '_batch' + str(batch_number) + '_dist.csv', delimiter=';')

            # Computes distances between correlation vectors.
            sim_corr_list = []
            for corr_vec in data_corr[:, 1:]:
                dist_corr_vec = np.linalg.norm(corr_vec - data_corr[:, 1:], ord=2, axis=1)
                sim_corr_list.append(dist_corr_vec)

            # Scales the distances to interval [0, 1].
            sim_corr = np.array(sim_corr_list)
            sim_corr = sim_corr / sim_corr.max()

            # Grouping based on weights' distances. Remember to also change the comparison operator below.
            # Weights' distance.
            # thresh = 0.6
            # groups_ds = data_dist[:, 1:]

            # Grouping based on correlation.
            thresh = 0.10
            groups_ds = sim_corr

            # Build the groups based on the threshold.
            groups = [set([i]) for i in range(groups_ds.shape[0])]
            for i in range(groups_ds.shape[0] - 1):
                for j in range(i + 1, groups_ds.shape[1]):

                    # Groups based on correlations.
                    # Compares both i,j and j,i because conditional entropy is not symmetric.
                    if groups_ds[i, j] <= thresh and groups_ds[j, i] <= thresh:

                        #         # Groups based on distance between learned weights.
                        #         if groups_ds[i, j] >= thresh:

                        for idx, g in enumerate(groups):
                            if i in g:
                                group_i = idx
                            if j in g:
                                group_j = idx
                        if group_i != group_j:
                            groups[group_i].update(groups[group_j])
                            del groups[group_j]
            print('Batch ' + str(batch_number))
            print(groups)
            # Eliminate duplicated groups.
            groups = list(set(frozenset(item) for item in groups))

            ds_attrs = {}
            ds_attrs['hospital_shuffled'] = ['Address1', 'City', 'Condition', 'CountyName', 'EmergencyService',
                                             'HospitalName', 'HospitalOwner', 'HospitalType', 'MeasureCode',
                                             'MeasureName', 'PhoneNumber', 'ProviderNumber', 'Sample', 'Score',
                                             'State', 'Stateavg', 'ZipCode']

            ds_attrs['food5k_shuffled'] = ['address', 'akaname', 'city', 'dbaname', 'facilitytype', 'inspectiondate',
                                           'inspectionid', 'inspectiontype', 'license', 'results', 'risk', 'state',
                                           'zip']

            ds_attrs['nypd6'] = ['ADDR_PCT_CD', 'BORO_NM', 'CRM_ATPT_CPTD_CD', 'JURISDICTION_CODE', 'JURIS_DESC',
                                 'KY_CD', 'LAW_CAT_CD', 'LOC_OF_OCCUR_DESC', 'OFNS_DESC', 'PATROL_BORO', 'PD_CD',
                                 'PD_DESC', 'PREM_TYP_DESC']

            ds_attrs['soccer'] = ['name', 'surname', 'birthyear', 'birthplace', 'position', 'team', 'city', 'stadium',
                                  'season', 'manager']

            print(groups)
            for group in groups:
                print([ds_attrs[self.feature_args['dataset_name']][i] for i in group])

            batch_number += 1

    def compute_KL(self):
        # This function compares current batch with the last training batch to try to skip training.

        batch_number = self.feature_args['first_batch']
        batch_size = self.feature_args['tuples_to_read_list'][batch_number - 1]
        self.setup_hc_repair_engine(batch_number, batch_size)

        smoothing_value = 0.0001
        smoothing_count = 0
        thresh = 0.015

        stats1 = None
        sum_freq1 = {}
        kl = {}

        for batch_size in self.feature_args['tuples_to_read_list'][self.feature_args['first_batch'] - 1:self.feature_args['last_batch']]:
            stats2_full = {}
            stats2_full['total_tuples'], stats2_full['single_attr_stats'], stats2_full['pair_attr_stats'] = self.load_stats(
                batch_number, batch_size)

            print('Batch: ' + str(batch_number))
            count_retrain = 0

            for attr1 in self.hc.ds.get_active_attributes():
                retrain = False
                kl[attr1] = {}
                # Computes the total frequency for attr1 in stats2 to normalize the frequencies.
                sum_freq2 = sum(stats2_full['single_attr_stats'][attr1].values())
                stats2 = stats2_full['pair_attr_stats']
                for attr2 in stats2[attr1].keys():
                    # Normalizes the frequencies in stats2 to get a valid data distribution.
                    for val1 in stats2[attr1][attr2].keys():
                        for val2 in stats2[attr1][attr2][val1].keys():
                            stats2[attr1][attr2][val1][val2] /= sum_freq2

                    if batch_number > self.feature_args['first_batch']:
                        # Smoothing of stats1. We assume that new values can appear (i.e., in stats2 but not in stats1)
                        # but no one is removed from the distribution (i.e., in stats1 but not in stats2).
                        stats1_attr1 = copy.deepcopy(stats1[attr1])
                        smoothing_count = 0
                        for val1 in stats2[attr1][attr2].keys():
                            # Adds to stats1 new values for attr1 in stats2 as well as all its co-occurring values.
                            if val1 not in stats1_attr1[attr2]:
                                stats1_attr1[attr2][val1] = {}
                                for val2 in stats2[attr1][attr2][val1].keys():
                                    stats1_attr1[attr2][val1][val2] = smoothing_value
                                    smoothing_count += 1
                            else:
                                # Adds to stats1 only new co-occurring values in stats2.
                                for val2 in stats2[attr1][attr2][val1].keys():
                                    if val2 not in stats1_attr1[attr2][val1]:
                                        stats1_attr1[attr2][val1][val2] = smoothing_value
                                        smoothing_count += 1

                        # Smoothing stats2. This is needed for stats on clean cells as cells can be spotted as errors
                        # in subsequent batches, so values in the distribution can disappear.
                        stats2_attr1 = copy.deepcopy(stats2[attr1])
                        smoothing_count2 = 0
                        for val1 in stats1_attr1[attr2].keys():
                            # Adds to stats2 removed values for attr1 in stats1 as well as all its co-occurring values.
                            if val1 not in stats2_attr1[attr2]:
                                stats2_attr1[attr2][val1] = {}
                                for val2 in stats1_attr1[attr2][val1].keys():
                                    stats2_attr1[attr2][val1][val2] = smoothing_value
                                    smoothing_count2 += 1
                            else:
                                # Adds to stats1 only new co-occurring values in stats2.
                                for val2 in stats1_attr1[attr2][val1].keys():
                                    if val2 not in stats2_attr1[attr2][val1]:
                                        stats2_attr1[attr2][val1][val2] = smoothing_value
                                        smoothing_count2 += 1

                        if smoothing_count > 0:
                            smoothing_correction = smoothing_value * (smoothing_count / sum_freq1[attr1])
                            for val1 in stats1_attr1[attr2].keys():
                                for val2 in stats1_attr1[attr2][val1].keys():
                                    # Smooths probabilities that do not correspond to new values.
                                    # We consider that no actual probability is equal to the smoothing value.
                                    if stats1_attr1[attr2][val1][val2] != smoothing_value:
                                        stats1_attr1[attr2][val1][val2] -= smoothing_correction

                        if smoothing_count2 > 0:
                            smoothing_correction2 = smoothing_value * (smoothing_count2 / sum_freq2)
                            for val1 in stats2_attr1[attr2].keys():
                                for val2 in stats2_attr1[attr2][val1].keys():
                                    # Smooths probabilities that do not correspond to new values.
                                    # We consider that no actual probability is equal to the smoothing value.
                                    if stats2_attr1[attr2][val1][val2] != smoothing_value:
                                        stats2_attr1[attr2][val1][val2] -= smoothing_correction2

                        # Generates the pdfs for (attr1, attr2) sorted by (val1, val2).
                        probs_attr1_stats1_list = []
                        probs_attr1_stats2_list = []
                        for val1 in sorted(stats1_attr1[attr2].keys()):
                            for val2 in sorted(stats1_attr1[attr2][val1].keys()):
                                if val1 != '_nan_' and val2 != '_nan_':
                                    probs_attr1_stats1_list.append(stats1_attr1[attr2][val1][val2])
                                    probs_attr1_stats2_list.append(stats2_attr1[attr2][val1][val2])

                        probs_attr1_stats1 = np.array(probs_attr1_stats1_list)
                        probs_attr1_stats2 = np.array(probs_attr1_stats2_list)

                        # Computes the K-L divergence.
                        kl[attr1][attr2] = entropy(probs_attr1_stats1, probs_attr1_stats2)

                        # Option 1) We retrain if at least one of the pairwise distributions regarding attr1 differs
                        # more than the threshold.
                        # if kl[attr1][attr2] > thresh:
                        #     retrain = True
                        #     print(str(batch_number) + ',' + attr1 + ',' + attr2 + ',' + str(kl[attr1][attr2]))

                    else:
                        # Saves the initial stats.
                        stats1 = stats2
                        sum_freq1[attr1] = sum_freq2
                        #     print(str(batch_number) + ',' + attr1 + ',' + attr2 + ',' + str(kl[attr1][attr2]))

                if batch_number > self.feature_args['first_batch']:
                    # Option 2) We retrain if the "aggregated" KL weighted by the attribute correlations is larger than
                    # the threshold multiplied by the number of attributes (to keep the threshold value proportional to
                    # each single attribute).
                    self.hc.ds.total_tuples, self.hc.ds.single_attr_stats, self.hc.ds.pair_attr_stats = \
                        stats2_full['total_tuples'], stats2_full['single_attr_stats'], stats2_full['pair_attr_stats']
                    correlations = self.hc.ds.compute_norm_cond_entropy_corr()

                    weighted_kl = [kl[attr1][attr2] * correlations[attr1][attr2] for attr2 in kl[attr1].keys()]
                    if sum(weighted_kl) / len(kl[attr1].values()) > thresh * len(kl[attr1].values()):
                        retrain = True
                        for attr2 in kl[attr1].keys():
                            print(str(batch_number) + ',' + attr1 + ',' + attr2 + ',' + str(kl[attr1][attr2]))

                if retrain:
                    count_retrain += 1
                    # Sets stats1 for attr1 to be the stats of the dataset used for retraining.
                    stats1[attr1] = stats2[attr1]
                    # Saves the new frequency of attr1.
                    sum_freq1[attr1] = sum_freq2
                    print('Updated stats1 for ' + attr1)

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
    # executor.generate_attr_corr_and_weight_dist()
    executor.get_attribute_groups()
    # executor.compute_KL()
