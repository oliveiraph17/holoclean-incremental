from examples.holoclean_incremental_repair_example import Executor

import os

hc_args = {
    'detectors': {'nulldetector': 'NullDetector',
                  'violationdetector': 'ViolationDetector'},
    'featurizers': {'occurattrfeat': 'OccurAttrFeaturizer'},
    'domain_thresh_1': 0,
    'weak_label_thresh': 0.99,
    'max_domain': 10000,
    'cor_strength': 0.6,
    'nb_cor_strength': 0.8,
    'epochs': 20,
    'threads': 1,
    'verbose': False,
    'timeout': 3*60000,
    'estimator_type': 'NaiveBayes',
    'current_iteration': None,
    'current_batch_number': None,
    'log_repairing_quality': True,
    'log_execution_times': False,
    'incremental': True,
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
    'approach': 'a',
    'tuples_to_read_list': [250] * 4,
    'number_of_iterations': 1,
}

executor = Executor(hc_args, inc_args)
executor.run()
