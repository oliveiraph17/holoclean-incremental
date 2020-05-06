import time

import numpy as np

import holoclean
import os
import logging
import sys

from repair.featurize.loadfeat_featurized_dataset import LoadFeatFeaturizedDataset


# noinspection PyPep8Naming,PyListCreation
class Executor:
    def __init__(self, hc_values, feature_values):
        self.hc = None
        self.hc_args = hc_values
        self.feature_args = feature_values
        self.training_cells = {}
        self.groups = None

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

        # Initializes the featurized dataset.
        self.hc.repair_engine.feat_dataset = LoadFeatFeaturizedDataset(self.hc.ds, self.hc.env, base_path)
        self.hc.repair_engine.feat_dataset.load_feat(batch_number, batch_size)

        # Sets attribute groups, if any.
        if self.groups is not None:
            self.hc.repair_engine.groups = self.groups

        # Sets up the models.
        self.hc.repair_engine.setup_repair_model()

        self.training_cells = {attr: 0 for attr in self.hc.ds.get_active_attributes()}

    # noinspection PyPep8Naming
    def train(self, batch_number=None):
        # Trains.
        tic = time.clock()
        total_training_cells = 0
        X_train, Y_train, mask_train, train_cid = \
            self.hc.repair_engine.feat_dataset.get_training_data(self.feature_args['detectors'],
                                                                 self.feature_args['labels'])
        for attr in self.hc.ds.get_active_attributes():
            self.training_cells[attr] = X_train[attr].size(0)
            logging.info('Training model for %s with %d training examples (cells)', attr, self.training_cells[attr])
            tic_attr = time.clock()

            self.hc.repair_engine.repair_model[attr].fit_model(X_train[attr], Y_train[attr], mask_train[attr],
                                                               self.hc.env['epochs'])

            if self.hc.env['save_load_checkpoint']:
                tic_checkpoint = time.clock()
                self.hc.repair_engine.repair_model[attr].save_checkpoint('/tmp/checkpoint-' + self.hc.ds.raw_data.name
                                                                         + '-' + attr)
                logging.debug("Checkpointing time %.2f.", time.clock() - tic_checkpoint)

            logging.info('Done. Elapsed time: %.2f', time.clock() - tic_attr)
            total_training_cells += self.training_cells[attr]
        toc = time.clock()

        logging.info('DONE training repair model')
        logging.debug('Time to fit repair model: %.2f secs', toc - tic)
        logging.debug('Number of training elements: %d', total_training_cells)

    # noinspection PyPep8Naming
    def train_grouped(self, retrain=None):
        # Trains.
        tic = time.clock()
        total_training_cells = 0
        X_train, Y_train, mask_train, train_cid = \
            self.hc.repair_engine.feat_dataset.get_training_data(self.feature_args['detectors'],
                                                                 self.feature_args['labels'])

        if retrain is None:
            # If no skipping list is provided, it retrains for all the groups.
            retrain = self.groups.keys()

        for attr, attrs_in_group in self.groups.items():
            if attr in retrain:
                self.training_cells[attr] = sum([X_train[att].size(0) for att in attrs_in_group])
                logging.info('Training model for %s with %d training examples (cells)', attr, self.training_cells[attr])
                tic_attr = time.clock()

                self.hc.repair_engine.repair_model[attr].fit_model_grouped(X_train, Y_train, mask_train,
                                                                   self.hc.env['epochs'], attrs_in_group)

                if self.hc.env['save_load_checkpoint']:
                    tic_checkpoint = time.clock()
                    self.hc.repair_engine.repair_model[attr].save_checkpoint('/tmp/checkpoint-'
                                                                             + self.hc.ds.raw_data.name
                                                                             + '-' + attr)
                    logging.debug("Checkpointing time %.2f.", time.clock() - tic_checkpoint)

                logging.info('Done. Elapsed time: %.2f', time.clock() - tic_attr)
                total_training_cells += self.training_cells[attr]
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
            if self.groups is None:
                Y_pred[attr] = self.hc.repair_engine.repair_model[attr].infer_values(X_pred[attr], mask_pred[attr])
            else:
                Y_pred[attr] = self.hc.repair_engine.repair_model[
                    self.hc.ds.get_attr_group(attr)].infer_values(X_pred[attr], mask_pred[attr])
            grdt = Y_truth[attr].numpy().flatten()
            Y_assign = Y_pred[attr].data.numpy().argmax(axis=1)
            accuracy = 100. * np.mean(Y_assign == grdt)
            logging.debug("Previous model accuracy: %.2f.", accuracy)

        logging.debug('Done. Elapsed time: %.2f', time.clock() - tic_attr)

        return Y_pred

    def evaluate(self, Y_pred, train_batch_number, skipping_batch_number, skipping=False):
        eval_metrics, incorrect_repairs = self.hc.repair_engine.feat_dataset.get_evaluation_metrics(Y_pred, skipping)

        metrics_header = ['infer_mode', 'features', 'train_using_all_batches', 'train_batch',
                          'infer_batch', 'dk_cells', 'training_cells', 'precision', 'recall', 'repairing_recall',
                          'f1', 'repairing_f1', 'detected_errors', 'total_errors', 'correct_repairs',
                          'total_repairs', 'total_repairs_grdt', 'total_repairs_grdt_correct',
                          'total_repairs_grdt_incorrect', 'rmse']

        fixed_metrics = {'infer_mode': self.hc.env['infer_mode'],
                         'features': 'global' if self.hc.env['global_features'] else 'incremental',
                         'train_using_all_batches': 'True',
                         'train_batch': train_batch_number,
                         'infer_batch': skipping_batch_number}

        agg_metrics = {name: 0 for name in eval_metrics.keys()}
        for attr in self.hc.ds.get_active_attributes():
            if self.groups is None:
                eval_metrics['training_cells'][attr] = self.training_cells[attr]
            else:
                eval_metrics['training_cells'][self.hc.ds.get_attr_group(attr)] = self.training_cells[
                    self.hc.ds.get_attr_group(attr)]

            # Outputs the attribute's metrics and adds them to the aggregated metrics.
            for metric, value in eval_metrics.items():
                # logging.debug('[' + attr + '] ' + metric + ' = %2.f', value[attr])

                # Sums for all aggregated metrics even though this is meaningless for some metrics because it is faster
                # than testing if they should be summed up or not.
                agg_metrics[metric] += value[attr]

            # Saves the quality metrics regarding the attribute.
            if self.hc_args['log_repairing_quality']:
                quality_log_file = self.quality_log_fpath + '_' + attr + '_quality_log.csv'
                with open(quality_log_file, 'a') as f:
                    # Writes the headers every time the model is trained as this starts a new repair iteration.
                    if not skipping:
                        f.write(';'.join(metric for metric in metrics_header) + '\n')

                    log_entry = []
                    for metric in metrics_header:
                        if metric in fixed_metrics:
                            log_entry.append(fixed_metrics[metric])
                        else:
                            log_entry.append(eval_metrics[metric][attr])

                    f.write(';'.join(str(value) for value in log_entry) + '\n')

            # Saves the mispredictions regarding the attribute.
            if self.hc_args['log_repairing_quality'] and attr in incorrect_repairs:
                mispredictions_log_file = self.quality_log_fpath + '_' + attr + '_mispredictions_log.csv'
                if not os.path.exists(mispredictions_log_file):
                    write_header = True
                else:
                    write_header = False

                with open(mispredictions_log_file, 'a') as f:
                    if write_header:
                        f.write('train_batch;infer_batch;tid;pred;truth;probs\n')
                    for row in incorrect_repairs[attr]:
                        f.write(str(train_batch_number) + ';' + str(skipping_batch_number) + ';' +
                                ';'.join(str(e) for e in row) + '\n')

            # Saves the weights of the attribute's model.
            if self.hc_args['log_feature_weights'] and not skipping:
                with open(self.weight_log_fpath + '_' + attr + '_weight_log.csv', 'a') as f:
                    # Writes the header in the first batch: [batch_number, w_0, w_1, ...].
                    if train_batch_number == 1:
                        # Gets the number of weights of the first model as all models have the same number of weights.
                        num_weights = next(iter(
                            self.hc.repair_engine.repair_model.values()
                        )).model.first_layer_weights[0].size(0)

                        f.write('batch_number;' + ';'.join('w' + str(i) for i in range(num_weights)) + '\n')

                    if self.groups is None:
                        weights = self.hc.repair_engine.repair_model[attr].model.first_layer_weights[0].squeeze().tolist()
                    else:
                        weights = self.hc.repair_engine.repair_model[
                            self.hc.ds.get_attr_group(attr)].model.first_layer_weights[0].squeeze().tolist()
                    f.write(str(train_batch_number) + ';' + ';'.join(str(w) for w in weights) + '\n')

        # Overwrites the aggregated metrics than are not only sums.
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

        # Outputs the aggregated metrics.
        # for metric, agg_value in agg_metrics.items():
        #     logging.debug('[Aggregated] ' + metric + ' = %.2f', agg_value)

        # Saves the aggregated metrics.
        if self.hc_args['log_repairing_quality']:
            log_entry = []
            for metric in metrics_header:
                if metric in fixed_metrics:
                    log_entry.append(fixed_metrics[metric])
                else:
                    log_entry.append(agg_metrics[metric])

            self.hc.experiment_quality_logger.info(';'.join(str(value) for value in log_entry))

    def run(self):
        # This function trains for each batch and infers from the batch onwards using the same set of trained models.
        batch_number = 1
        number_of_batches = len(self.feature_args['tuples_to_read_list'])

        tic = time.time()

        for batch_size in self.feature_args['tuples_to_read_list']:
            if batch_number > self.feature_args['last_batch']:
                break

            if batch_number >= self.feature_args['first_batch']:
                self.setup_hc_repair_engine(batch_number, batch_size)
                if self.groups is None:
                    self.train()
                else:
                    self.train_grouped()
                Y_pred = self.infer()
                self.evaluate(Y_pred, batch_number, batch_number)

                number_of_skipping_batches = number_of_batches - batch_number
                for i in range(number_of_skipping_batches):
                    self.hc.repair_engine.feat_dataset.load_feat_skipping(batch_number, batch_size, i + 1)
                    Y_pred = self.infer(skipping=True)
                    self.evaluate(Y_pred, batch_number, batch_number + i + 1, skipping=True)

            batch_number += 1

        toc = time.time()
        print('Elapsed time: ', toc - tic)

    def run_infer_all(self, train_batch):
        # This function loads a given trained set of models (in /tmp folder) and infers all the batches using them.
        batch_size = self.feature_args['tuples_to_read_list'][0]
        self.hc_args['save_load_checkpoint'] = True

        # We set up using batch 2 because it consumes the least memory possible being the first loaded from disk (the
        # first batch is not loaded from /tmp by default). Actually, it loads the "given" trained model from /tmp.
        # That is, the batch number here is meaningless for training issues.
        self.setup_hc_repair_engine(2, batch_size)

        # We do not train because we use save_load_checkpoint. This is just for logging purposes.
        for attr in self.hc.ds.get_active_attributes():
            self.training_cells[attr] = 0

        for i in range(self.feature_args['first_batch'], self.feature_args['last_batch'] + 1):
            self.hc.repair_engine.feat_dataset.load_feat_skipping(0, batch_size, i)
            Y_pred = self.infer(skipping=True)
            self.evaluate(Y_pred, train_batch_number=train_batch, skipping_batch_number=i, skipping=True)

    def run_skipping_predefined(self, corr_grouping_thresh=None):
        from examples import statistics_calculations

        # Creates a stats Executor using the same args
        stats_executor = statistics_calculations.Executor(hc_args, feature_args)

        # This function trains only at predefined batches and skips training for the other ones.
        batch_number = 1
        number_of_batches = len(self.feature_args['tuples_to_read_list'])
        self.hc_args['save_load_checkpoint'] = True

        # It is possible to set the groups either manually (as follows), or automatically by setting the threshold.
        # # Single model for all the attributes
        # self.groups = {
        #         'Address1': ['Address1', 'City', 'CountyName', 'HospitalName', 'MeasureCode', 'MeasureName',
        #                      'PhoneNumber', 'ProviderNumber', 'Sample', 'Stateavg', 'ZipCode', 'Condition',
        #                      'State', 'HospitalType', 'HospitalOwner', 'Score', 'EmergencyService'],
        # }

        # # One model per attribute.
        # self.groups = {
        #     'Address1': ['Address1'], 'City': ['City'], 'CountyName': ['CountyName'],
        #     'HospitalName': ['HospitalName'], 'PhoneNumber': ['PhoneNumber'], 'ProviderNumber': ['ProviderNumber'],
        #     'Sample': ['Sample'], 'ZipCode': ['ZipCode'], 'Condition': ['Condition'], 'State': ['State'],
        #     'MeasureCode': ['MeasureCode'], 'MeasureName': ['MeasureName'], 'Stateavg': ['Stateavg'],
        #     'HospitalType': ['HospitalType'], 'HospitalOwner': ['HospitalOwner'], 'Score': ['Score'],
        #     'EmergencyService': ['EmergencyService'],
        # }

        # # Manual setting of groups according to a previous run of stats_calc.run_get_attribute_groups (thresh=0.1)
        # if batch_number == 1:
        #     self.groups = {
        #         'Address1': ['Address1', 'City', 'CountyName', 'HospitalName', 'MeasureCode', 'MeasureName',
        #                      'PhoneNumber', 'ProviderNumber', 'Sample', 'Stateavg', 'ZipCode'],
        #         'Condition': ['Condition'],
        #         'State': ['State'],
        #         'HospitalType': ['HospitalType'],
        #         'HospitalOwner': ['HospitalOwner'],
        #         'Score': ['Score'],
        #         'EmergencyService': ['EmergencyService'],
        #     }
        # else:
        #     self.groups = {
        #         'Address1': ['Address1', 'City', 'CountyName', 'HospitalName', 'PhoneNumber', 'ProviderNumber',
        #                      'Sample', 'ZipCode'],
        #         'Condition': ['Condition'],
        #         'State': ['State'],
        #         'MeasureCode': ['MeasureCode', 'MeasureName', 'Stateavg'],
        #         'HospitalType': ['HospitalType'],
        #         'HospitalOwner': ['HospitalOwner'],
        #         'Score': ['Score'],
        #         'EmergencyService': ['EmergencyService'],
        #     }

        tic = time.time()

        for batch_size in self.feature_args['tuples_to_read_list']:
            if batch_number > self.feature_args['last_batch']:
                break

            if batch_number >= self.feature_args['first_batch']:
                # Trains only at predefined batches.
                if batch_number in self.feature_args['train_batches']:
                    if corr_grouping_thresh is not None:
                        self.groups = stats_executor.get_attribute_groups(batch_number, thresh=corr_grouping_thresh)
                    self.setup_hc_repair_engine(batch_number, batch_size)
                    if self.groups is None:
                        self.train()
                    else:
                        self.train_grouped()

                else:
                    self.setup_hc_repair_engine(batch_number, batch_size)

                Y_pred = self.infer()
                self.evaluate(Y_pred, batch_number, batch_number)

            batch_number += 1

        toc = time.time()
        print('Elapsed time: ', toc - tic)

    def run_skipping_KL(self, corr_grouping_thresh, kl_thresh):
        from examples import statistics_calculations

        # Creates a stats Executor using the same args
        stats_executor = statistics_calculations.Executor(hc_args, feature_args)

        # Pair stats PDF and attribute frequencies for the previous stats.
        stats = None
        sum_freq = {}

        # List of groups that should be retrained.
        retrain = None

        batch_number = 1
        number_of_batches = len(self.feature_args['tuples_to_read_list'])
        self.hc_args['save_load_checkpoint'] = True

        tic = time.time()

        for batch_size in self.feature_args['tuples_to_read_list']:
            if batch_number > self.feature_args['last_batch']:
                break

            # self.groups = {
            #     'Address1': ['Address1'], 'City': ['City'], 'CountyName': ['CountyName'],
            #     'HospitalName': ['HospitalName'], 'PhoneNumber': ['PhoneNumber'], 'ProviderNumber': ['ProviderNumber'],
            #     'Sample': ['Sample'], 'ZipCode': ['ZipCode'], 'Condition': ['Condition'], 'State': ['State'],
            #     'MeasureCode': ['MeasureCode'], 'MeasureName': ['MeasureName'], 'Stateavg': ['Stateavg'],
            #     'HospitalType': ['HospitalType'], 'HospitalOwner': ['HospitalOwner'], 'Score': ['Score'],
            #     'EmergencyService': ['EmergencyService'],
            # }
            self.groups = stats_executor.get_attribute_groups(batch_number, thresh=corr_grouping_thresh)

            self.setup_hc_repair_engine(batch_number, batch_size)
            stats_executor.setup_hc_repair_engine(batch_number, batch_size)

            if batch_number == self.feature_args['first_batch']:
                # Sets all the groups to be trained in the first batch.
                retrain = [att_rep for att_rep in self.groups.keys()]

                # Gets the stats for providing in upcoming KL-based skipping calls.
                stats_full = {}
                stats_full['total_tuples'], stats_full['single_attr_stats'], stats_full[
                    'pair_attr_stats'] = stats_executor.load_stats(
                    batch_number, batch_size)
                stats = stats_full['pair_attr_stats']

                for attr in stats.keys():
                    sum_freq[attr] = sum(stats_full['single_attr_stats'][attr].values())

                for attr1 in stats.keys():
                    for attr2 in stats[attr1].keys():
                        # Normalizes the frequencies in stats2 to get a valid data distribution.
                        for val1 in stats[attr1][attr2].keys():
                            for val2 in stats[attr1][attr2][val1].keys():
                                stats[attr1][attr2][val1][val2] /= sum_freq[attr1]

            else:
                # Gets the output of KL for the next batches.
                _, stats, sum_freq, retrain = stats_executor.compute_KL(batch_size=batch_size,
                                                                        batch_number=batch_number, stats1=stats,
                                                                        sum_freq1=sum_freq, thresh=kl_thresh,
                                                                        groups=self.groups)

            print('Batch: ' + str(batch_number))
            print(self.groups)
            print(retrain)

            # Skip training based on the retrain list.
            self.train_grouped(retrain=retrain)

            Y_pred = self.infer()
            self.evaluate(Y_pred, batch_number, batch_number)

            batch_number += 1

        toc = time.time()
        print('Elapsed time: ', toc - tic)


if __name__ == "__main__":
    # Default parameters for HoloClean.
    hc_args = {
        'epochs': 20,
        'threads': 1,
        'verbose': True,
        'print_fw': True,
        'timeout': 3 * 60000,
        'incremental': False,
        'infer_mode': sys.argv[2],
        'global_features': bool(int(sys.argv[3])),
        'log_repairing_quality': True,
        'log_execution_times': False,
        'log_feature_weights': True,
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
        'labels': 'weak',  # one of 'weak', 'init' or 'truth'
        'detectors': ['NullDetector', 'ViolationDetector'],  # ['NullDetector', 'ViolationDetector', 'ErrorLoaderDetector'],
        'first_batch': int(sys.argv[5]),
        'last_batch': int(sys.argv[6]),
        'train_batches': [1, 2, 4, 8, 16, 32, 64],  # PG
        # 'train_batches': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # PA
    }

    # Runs the default example.
    executor = Executor(hc_args, feature_args)
    # executor.run()

    # Runs the run_infer_all
    # 1) Set save_load_checkpoint=True
    # 2) Run for first_batch=1 and last_batch=1 to quickly generate checkpoints
    # 3) Run for first_batch=100 and last_batch=100 to generate the checkpoint referring to the full dataset
    # 4) Back up the checkpoint to avoid being overwritten
    # 5) Run first_batch=1 and last_batch=100 after commenting executor.run() above and uncommenting the function below
    # executor.run_infer_all(train_batch=100)

    # Runs the option for skipping at predefined batches (train_batches), optionally setting groups manually.
    # executor.run_skipping_predefined(corr_grouping_thresh=0.05)

    # Runs the option for automatic grouping and KL-based skipping.
    executor.run_skipping_KL(corr_grouping_thresh=0.05, kl_thresh=0.01)
