from examples.holoclean_incremental_repair_example import Executor

import os

hc_args = {
    'detectors': {'nulldetector': 'NullDetector', 'violationdetector': 'ViolationDetector'},
    'featurizers': {'occurattrfeat': 'OccurAttrFeaturizer'},
    'domain_thresh_1': 0,
    'weak_label_thresh': 0.99,
    'max_domain': 50,
    'cor_strength': 0.6,
    'nb_cor_strength': 0.8,
    'epochs': 20,
    'threads': 1,
    'verbose': False,
    'timeout': 3*60000,
    'estimator_type': 'NaiveBayes',
    'epochs_convergence': 3,
    'convergence_thresh': 0.01,
    'current_iteration': None,
    'current_batch_number': None,
    'log_repairing_quality': True,
    'log_execution_times': True,
    'incremental': None,
    'incremental_entropy': False,
    'default_entropy': False,
    'repair_previous_errors': None,
    'recompute_from_scratch': None,
    'skip_training': False,
    'ignore_previous_training_cells': False,
    'save_load_checkpoint': False,
    'append': True
}

inc_args = {
    'project_root': os.environ['HOLOCLEANHOME'],
    'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
    'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
    'dataset_name': None,
    'entity_col': None,
    'approach': 'co_a',
    'tuples_to_read_list': [250] * 4,
    'iterations': [0],
}

######################################################################
# Co-occurrence featurizer
hc_args['featurizers'] = {'occurattrfeat': 'OccurAttrFeaturizer'}
hc_args['estimator_type'] = 'NaiveBayes'
######################################################################

datasets = {'food5k_shuffled': '_tid_'}

for dataset_name, entity_col in datasets.items():
    inc_args['dataset_name'] = dataset_name
    inc_args['entity_col'] = entity_col

    ######################################################################

    hc_args['incremental'] = False
    hc_args['repair_previous_errors'] = False
    hc_args['recompute_from_scratch'] = False

    # A - Quality
    hc_args['log_repairing_quality'] = True
    hc_args['log_execution_times'] = True
    executor = Executor(hc_args, inc_args)
    executor.run()

    # A - Time
    hc_args['log_repairing_quality'] = False
    hc_args['log_execution_times'] = True
    inc_args['iterations'] = [1, 2]
    executor = Executor(hc_args, inc_args)
    executor.run()

    ######################################################################

    hc_args['incremental'] = True
    hc_args['repair_previous_errors'] = True
    hc_args['recompute_from_scratch'] = True
    inc_args['approach'] = 'co_b'

    # B - Quality
    hc_args['log_repairing_quality'] = True
    hc_args['log_execution_times'] = True
    inc_args['iterations'] = [0]
    executor = Executor(hc_args, inc_args)
    executor.run()

    # B - Time
    hc_args['log_repairing_quality'] = False
    hc_args['log_execution_times'] = True
    inc_args['iterations'] = [1, 2]
    executor = Executor(hc_args, inc_args)
    executor.run()

    ######################################################################

    hc_args['incremental'] = True
    hc_args['repair_previous_errors'] = False
    hc_args['recompute_from_scratch'] = False
    inc_args['approach'] = 'co_c'

    # C - Quality
    hc_args['log_repairing_quality'] = True
    hc_args['log_execution_times'] = True
    inc_args['iterations'] = [0]
    executor = Executor(hc_args, inc_args)
    executor.run()

    # C - Time
    hc_args['log_repairing_quality'] = False
    hc_args['log_execution_times'] = True
    inc_args['iterations'] = [1, 2]
    executor = Executor(hc_args, inc_args)
    executor.run()

    ######################################################################

    hc_args['incremental'] = True
    hc_args['repair_previous_errors'] = True
    hc_args['recompute_from_scratch'] = True
    hc_args['save_load_checkpoint'] = True
    inc_args['approach'] = 'co_bplus'

    # B+ - Quality
    hc_args['log_repairing_quality'] = True
    hc_args['log_execution_times'] = True
    inc_args['iterations'] = [0]
    executor = Executor(hc_args, inc_args)
    executor.run()

    # B+ - Time
    hc_args['log_repairing_quality'] = False
    hc_args['log_execution_times'] = True
    inc_args['iterations'] = [1, 2]
    executor = Executor(hc_args, inc_args)
    executor.run()

    ######################################################################

    hc_args['incremental'] = True
    hc_args['repair_previous_errors'] = False
    hc_args['recompute_from_scratch'] = False
    hc_args['save_load_checkpoint'] = True
    inc_args['approach'] = 'co_cplus'

    # C+ - Quality
    hc_args['log_repairing_quality'] = True
    hc_args['log_execution_times'] = True
    inc_args['iterations'] = [0]
    executor = Executor(hc_args, inc_args)
    executor.run()

    # C+ - Time
    hc_args['log_repairing_quality'] = False
    hc_args['log_execution_times'] = True
    inc_args['iterations'] = [0, 1, 2]
    executor = Executor(hc_args, inc_args)
    executor.run()
