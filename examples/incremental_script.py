from examples.holoclean_incremental_repair_example import Executor

import os

hc_args = {
    'detectors': {'nulldetector': 'NullDetector', 'violationdetector': 'ViolationDetector'},
    'featurizers': {'occurattrfeat': 'OccurAttrFeaturizer'},
    'domain_thresh_1': 0,
    'weak_label_thresh': 0.60,
    'max_domain': 1000,
    'cor_strength': 0.2,
    'nb_cor_strength': 0.3,
    'epochs': 50,
    'threads': 1,
    'verbose': False,
    'timeout': 3*60000,
    'estimator_type': 'NaiveBayes',
    'epochs_convergence': 5,
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
    'tuples_to_read_list': [1000] * 5,
    'iterations': [0],
}

######################################################################
# Co-occurrence featurizer
hc_args['featurizers'] = {'occurattrfeat': 'OccurAttrFeaturizer'}
hc_args['estimator_type'] = 'NaiveBayes'
######################################################################

datasets = []
datasets.append(('hospital', None, [20] * 2, 0.99, 10000, 0.6, 0.8))
datasets.append(('food5k_shuffled', '_tid_', [1000] * 5, 0.6, 1000, 0.2, 0.3))

approaches = ['A', 'B', 'C', 'C+', 'B+', 'Full']
avg_time_iterations = [1, 2]  # or None

for dataset_name, entity_col, tuples_to_read_list, weak_label_thresh, max_domain, cor_strength, nb_cor_strength in datasets:
    inc_args['dataset_name'] = dataset_name
    inc_args['entity_col'] = entity_col
    inc_args['tuples_to_read_list'] = tuples_to_read_list
    hc_args['weak_label_thresh'] = weak_label_thresh
    hc_args['max_domain'] = max_domain
    hc_args['cor_strength'] = cor_strength
    hc_args['nb_cor_strength'] = nb_cor_strength

    ######################################################################

    if 'A' in approaches:
        hc_args['incremental'] = False
        hc_args['repair_previous_errors'] = False
        hc_args['recompute_from_scratch'] = False
        inc_args['approach'] = 'co_a'

        # A - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [0]
        executor = Executor(hc_args, inc_args)
        executor.run()

        if avg_time_iterations:
            # A - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            executor = Executor(hc_args, inc_args)
            executor.run()

    ######################################################################

    if 'B' in approaches:
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

        if avg_time_iterations:
            # B - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            executor = Executor(hc_args, inc_args)
            executor.run()

    ######################################################################

    if 'C' in approaches:
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

        if avg_time_iterations:
            # C - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            executor = Executor(hc_args, inc_args)
            executor.run()

    ######################################################################

    if 'B+' in approaches:
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

        if avg_time_iterations:
            # B+ - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            executor = Executor(hc_args, inc_args)
            executor.run()

    ######################################################################

    if 'C+' in approaches:
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

        if avg_time_iterations:
            # C+ - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            executor = Executor(hc_args, inc_args)
            executor.run()

    ######################################################################

    if 'Full' in approaches:
        hc_args['incremental'] = False
        hc_args['repair_previous_errors'] = False
        hc_args['recompute_from_scratch'] = False
        hc_args['save_load_checkpoint'] = False
        inc_args['approach'] = 'co_full'

        # Full - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [0]
        executor = Executor(hc_args, inc_args)
        executor.run()

        if avg_time_iterations:
            # Full - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            executor = Executor(hc_args, inc_args)
            executor.run()
