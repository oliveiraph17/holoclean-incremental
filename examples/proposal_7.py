import holoclean
import logging
import sys
import os
from detect import NullDetector, ViolationDetector
from repair.featurize import *
sys.path.append('../')

dataset_name = 'hospital'
batches = ['0001-0100', '0101-0200', '0201-0300', '0301-0400', '0401-0500',
           '0501-0600', '0601-0700', '0701-0800', '0801-0900', '0901-1000']
# batches = ['0001-0500', '0501-1000']

drop = 'y'

# We may run out of memory if HoloClean is not reinstantiated at each loading step.
for batch in batches:
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
        incremental=True,
        incremental_entropy=False,
        default_entropy=False,
        repair_previous_errors=False,
        recompute_from_scratch=False,
        skip_training=False,
        ignore_previous_training_cells=False,
        save_load_checkpoint=True
    ).session

    if batch == batches[0]:
        if drop == 'y':
            table_list = [dataset_name + '_repaired', 'single_attr_stats', 'pair_attr_stats', 'training_cells']
            hc.ds.engine.drop_tables(table_list)
            if os.path.exists('/tmp/checkpoint.tar'):
                os.remove('/tmp/checkpoint.tar')

    # Load existing data and Denial Constraints.
    hc.load_data(dataset_name, '../testdata/' + dataset_name + '_' + batch + '.csv')
    hc.load_dcs('../testdata/' + dataset_name + '_constraints.txt')
    hc.ds.set_constraints(hc.get_dcs())

    # Detect erroneous cells using these two detectors.
    detectors = [NullDetector(), ViolationDetector()]
    hc.detect_errors(detectors)

    # Repair errors based on the defined features.
    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        ConstraintFeaturizer()
    ]
    hc.repair_errors(featurizers)

    # Evaluate the correctness of the results.
    hc.evaluate(fpath='../testdata/' + dataset_name + '_clean.csv',
                tid_col='tid',
                attr_col='attribute',
                val_col='correct_val')

    logging.info('[EXECUTION_TIME] Batch %s finished.', batch)
