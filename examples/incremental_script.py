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
    'incremental': False,
    'incremental_entropy': False,
    'default_entropy': False,
    'repair_previous_errors': False,
    'recompute_from_scratch': False,
    'skip_training': False,
    'ignore_previous_training_cells': False,
    'save_load_checkpoint': False,
}

inc_args = {
    'project_root': os.environ['HOLOCLEANHOME'],
    'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
    'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
    'dataset_name': 'hospital',
    'approach': 'co_a',
    'tuples_to_read_list': [250] * 4,
    'iterations': [0],
}

######################################################################
# Co-occurrence featurizer
hc_args['featurizers'] = {'occurattrfeat': 'OccurAttrFeaturizer'}
hc_args['estimator_type'] = 'NaiveBayes'
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

inc_args['dataset_dir'] = os.environ['HOLOCLEANHOME'] + '/testdata/confidential/'
inc_args['log_dir'] = os.environ['HOLOCLEANHOME'] + '/experimental_results/confidential/'
inc_args['dataset_name'] = 'adult_1'
inc_args['tuples_to_read_list'] = [2000] * 4
inc_args['iterations'] = [0]

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
