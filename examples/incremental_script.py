from examples.holoclean_incremental_repair_example import Executor

import os

# Pattern => file_name: class_name
detectors = [
    {'nulldetector': 'NullDetector',
     'violationdetector': 'ViolationDetector'},

    {'errorloaderdetector': 'ErrorsLoaderDetector'}
]

# Pattern => file_name: class_name
featurizers = [
    {'occurattrfeat': 'OccurAttrFeaturizer'},

    {'occurattrfeat': 'OccurAttrFeaturizer',
     'freqfeat': 'FreqFeaturizer',
     'constraintfeat': 'ConstraintFeaturizer'},

    {'embeddingfeat': 'EmbeddingFeaturizer'},

    {'embeddingfeat': 'EmbeddingFeaturizer',
     'constraintfeat': 'ConstraintFeaturizer'}
]

# Pattern => file_name: class_name
featurizers_error_loader = [
    {'occurattrfeat': 'OccurAttrFeaturizer'},

    {'occurattrfeat': 'OccurAttrFeaturizer',
     'freqfeat': 'FreqFeaturizer'},

    {'embeddingfeat': 'EmbeddingFeaturizer'}
]

domain_thresh_1_values = [0, 0.1, 0.2, 0.3, 0.4]

estimator_types = ['NaiveBayes', 'Logistic', 'TupleEmbedding']

hc_args = {
    'detectors': None,
    'featurizers': None,
    'domain_thresh_1': 0,
    'weak_label_thresh': 0.99,
    'max_domain': 10000,
    'cor_strength': 0.6,
    'nb_cor_strength': 0.8,
    'epochs': 2,
    'threads': 1,
    'verbose': False,
    'timeout': 3*60000,
    'estimator_type': None,
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
    'tuples_to_read_list': [100] * 10,
    'number_of_iterations': 1,
}

executor = Executor(hc_args, inc_args)
executor.run()
