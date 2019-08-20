import holoclean
import logging
import os
from detect import NullDetector, ViolationDetector
from repair.featurize import *

dataset_dir = '../testdata/hospital/'
dataset_name = 'hospital'
# batches = ['0001-0100', '0101-0200', '0201-0300', '0301-0400', '0401-0500',
#            '0501-0600', '0601-0700', '0701-0800', '0801-0900', '0901-1000']
batches = ['0001-0500', '0501-1000']

number_of_iterations = 1

log_repairing_quality = True
log_execution_times = False
log_fpath = ''

if log_repairing_quality:
    log_fpath += '/home/ph/Git/HoloClean/experiments/' + dataset_name + '/2_batches/repairing_quality/proposal_4.csv'

if log_execution_times:
    log_fpath += '/home/ph/Git/HoloClean/experiments/' + dataset_name + '/2_batches/execution_times/proposal_4.csv'

for current_iteration in range(number_of_iterations):
    current_batch_number = 0

    # We may run out of memory if HoloClean is not reinstantiated at each loading step.
    for current_batch in batches:
        # Sets up a HoloClean session.
        hc = holoclean.HoloClean(
            db_name='holo',
            domain_thresh_1=0,
            domain_thresh_2=0,
            weak_label_thresh=0.99,
            max_domain=10000,
            cor_strength=0.6,
            nb_cor_strength=0.8,
            epochs=10,
            weight_decay=0.01,
            learning_rate=0.001,
            threads=1,
            batch_size=1,
            verbose=False,
            timeout=3*60000,
            feature_norm=False,
            weight_norm=False,
            print_fw=False,
            current_batch_number=current_batch_number,
            current_iteration=current_iteration,
            log_repairing_quality=log_repairing_quality,
            log_execution_times=log_execution_times,
            incremental=True,
            incremental_entropy=False,
            default_entropy=False,
            repair_previous_errors=True,
            recompute_from_scratch=False,
            skip_training=False,
            ignore_previous_training_cells=False,
            save_load_checkpoint=True
        ).session

        # Drops tables and model checkpoint in the first batch.
        if current_batch == batches[0]:
            table_list = [dataset_name + '_repaired',
                          'single_attr_stats',
                          'pair_attr_stats',
                          'training_cells',
                          'repaired_table_copy']

            hc.ds.engine.drop_tables(table_list)

            if os.path.exists('/tmp/checkpoint.tar'):
                os.remove('/tmp/checkpoint.tar')

        # Sets up logger for the experiments.
        if log_repairing_quality:
            hc.setup_experiment_logger('repairing_quality_logger', log_fpath)
        elif log_execution_times:
            hc.setup_experiment_logger('execution_time_logger', log_fpath)

        # Loads existing data and Denial Constraints.
        hc.load_data(dataset_name, dataset_dir + dataset_name + '_' + current_batch + '.csv')
        hc.load_dcs(dataset_dir + dataset_name + '_constraints.txt')
        hc.ds.set_constraints(hc.get_dcs())

        # Detects erroneous cells using these two detectors.
        detectors = [NullDetector(), ViolationDetector()]
        hc.detect_errors(detectors)

        # Repairs errors based on the defined features.
        hc.setup_domain()
        featurizers = [
            InitAttrFeaturizer(),
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer()
        ]
        hc.repair_errors(featurizers)

        # Evaluates the correctness of the results.
        hc.evaluate(fpath=dataset_dir + dataset_name + '_clean.csv',
                    tid_col='tid',
                    attr_col='attribute',
                    val_col='correct_val')

        logging.info('Batch %s finished.', current_batch)
        current_batch_number += 1
