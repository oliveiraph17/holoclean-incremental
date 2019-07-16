import holoclean
import logging
import sys
from detect import NullDetector, ViolationDetector
from repair.featurize import *
sys.path.append('../')

dataset_name = 'hospital'
# batches = ['1-100', '101-200', '201-300', '301-400', '401-500',
#            '501-600', '601-700', '701-800', '801-900', '901-1000']
batches = ['1-900', '901-1000']

# This line pauses the execution to drop the tables if needed.
drop = None
while drop != 'y' and drop != 'n':
    drop = input('Do you want to drop tables <dataset>_repaired, single_attr_stats, and pair_attr_stats? (y/n)\n')

# We may run out of memory if HoloClean is not reinstantiated at each loading step.
for batch in batches:
    # Setup a HoloClean session.
    hc = holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=2,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=False,
        timeout=3 * 60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=False,
        incremental=True,
        incremental_entropy=False
    ).session

    if batch == batches[0]:
        if drop == 'y':
            hc.ds.engine.drop_tables([dataset_name + '_repaired', 'single_attr_stats', 'pair_attr_stats'])

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

    logging.info('Batch %s finished', batch)
