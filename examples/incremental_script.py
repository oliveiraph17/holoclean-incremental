from examples.holoclean_incremental_repair_example import Executor

import os

hc_args = {
    # 'detectors': [('nulldetector', 'NullDetector', False),
    #               ('violationdetector', 'ViolationDetector', False)],
    'detectors': [('errorloaderdetector', 'ErrorsLoaderDetector', True)],
    'featurizers': {'occurattrfeat': 'OccurAttrFeaturizer'},
    'domain_thresh_1': 0,
    'domain_thresh_2': 0,
    'weak_label_thresh': 0.99,
    'max_domain': 10000,
    'cor_strength': 0.6,
    'nb_cor_strength': 0.8,
    'epochs': 20,
    'threads': 1,
    'verbose': True,
    'timeout': 3 * 60000,
    'estimator_type': 'NaiveBayes',
    'epochs_convergence': 3,
    'convergence_thresh': 0.01,
    'skip_training_thresh': 101,
    'log_repairing_quality': True,
    'log_execution_times': True,
    'log_feature_weights': True,
    'incremental': None,
    'repair_previous_errors': None,
    'recompute_from_scratch': None,
    'save_load_checkpoint': None,
    'append': None,
    'infer_mode': None,
    'global_features': None,
    'train_using_all_batches': None
}

inc_args = {
    'project_root': os.environ['HOLOCLEANHOME'],
    'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
    'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
    'dataset_name': None,
    'entity_col': None,
    'numerical_attrs': None,
    'tuples_to_read_list': None,
    'model_monitoring': False,
    'dataset_size': None,
    'dataset_fraction_for_batch': None,
    'skip_training_starting_batch': -1,
    'iterations': [1]
}


def build_model_monitoring_input(total_tuples, percentage):
    tuples_to_read_skipping_dict = {}

    batch_size = int(total_tuples * percentage)
    number_of_batches = int(total_tuples / batch_size)

    for i in range(number_of_batches - 1):
        tuples_to_read_skipping_dict[i + 2] = [batch_size * (i + 1)]
        for j in range(number_of_batches - (i + 1)):
            tuples_to_read_skipping_dict[i + 2].append(batch_size)

    return tuples_to_read_skipping_dict


datasets = [
    ##############################
    # Without global features
    ##############################

    ('hospital_numerical', None, ['Score', 'Sample'], True,
     [(100, ['Score']), (150, ['Sample'])],
     None, True, 1000, 0.02, 'NaiveBayes',
     0.99, 10000, 0.6, 0.8,
     'dk', False, False),

    ('hospital_numerical', None, ['Score', 'Sample'], True,
     [(100, ['Score']), (150, ['Sample'])],
     None, True, 1000, 0.02, 'NaiveBayes',
     0.99, 10000, 0.6, 0.8,
     'all', False, False),

    ('hospital_numerical', None, ['Score', 'Sample'], True,
     [(100, ['Score']), (150, ['Sample'])],
     None, True, 1000, 0.10, 'NaiveBayes',
     0.99, 10000, 0.6, 0.8,
     'dk', False, False),

    ('hospital_numerical', None, ['Score', 'Sample'], True,
     [(100, ['Score']), (150, ['Sample'])],
     None, True, 1000, 0.10, 'NaiveBayes',
     0.99, 10000, 0.6, 0.8,
     'all', False, False),

    ('hospital_numerical', None, ['Score', 'Sample'], True,
     [(100, ['Score']), (150, ['Sample'])],
     None, True, 1000, 0.20, 'NaiveBayes',
     0.99, 10000, 0.6, 0.8,
     'dk', False, False),

    ('hospital_numerical', None, ['Score', 'Sample'], True,
     [(100, ['Score']), (150, ['Sample'])],
     None, True, 1000, 0.20, 'NaiveBayes',
     0.99, 10000, 0.6, 0.8,
     'all', False, False),

    # ('food5k', None, ['latitude', 'longitude'], True,
    #  [(100, ['latitude', 'longitude'])],
    #  None, True, 5000, 0.01, 'NaiveBayes',
    #  0.6, 1000, 0.2, 0.3,
    #  'dk', False, False),
    #
    # ('food5k', None, ['latitude', 'longitude'], True,
    #  [(100, ['latitude', 'longitude'])],
    #  None, True, 5000, 0.01, 'NaiveBayes',
    #  0.6, 1000, 0.2, 0.3,
    #  'all', False, False),
    #
    # ('food5k_shuffled', '_tid_', ['latitude', 'longitude'], True,
    #  [(100, ['latitude', 'longitude'])],
    #  None, True, 5000, 0.01, 'NaiveBayes',
    #  0.6, 1000, 0.2, 0.3,
    #  'dk', False, False),
    #
    # ('food5k_shuffled', '_tid_', ['latitude', 'longitude'], True,
    #  [(100, ['latitude', 'longitude'])],
    #  None, True, 5000, 0.01, 'NaiveBayes',
    #  0.6, 1000, 0.2, 0.3,
    #  'all', False, False),
    #
    # ('nypd6', None, ['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude'], True,
    #  [(100, ['X_COORD_CD', 'Y_COORD_CD']), (100, ['Latitude', 'Longitude'])],
    #  None, True, 32400, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'dk', False, False),
    #
    # ('nypd6', None, ['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude'], True,
    #  [(100, ['X_COORD_CD', 'Y_COORD_CD']), (100, ['Latitude', 'Longitude'])],
    #  None, True, 32400, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'all', False, False),
    #
    # ('nypd6_shuffled', '_tid_', ['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude'], True,
    #  [(100, ['X_COORD_CD', 'Y_COORD_CD']), (100, ['Latitude', 'Longitude'])],
    #  None, True, 32400, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'dk', False, False),
    #
    # ('nypd6_shuffled', '_tid_', ['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude'], True,
    #  [(100, ['X_COORD_CD', 'Y_COORD_CD']), (100, ['Latitude', 'Longitude'])],
    #  None, True, 32400, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'all', False, False),
    #
    # ('soccer', None, None, False,
    #  None,
    #  None, True, 200000, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'dk', False, False),
    #
    # ('soccer', None, None, False,
    #  None,
    #  None, True, 200000, 0.01,
    #  0.9, 100, 0.05, 0.3,
    #  'all', False, False),
    #
    # ('soccer_shuffled', '_tid_', None, False,
    #  None,
    #  None, True, 200000, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'dk', False, False),
    #
    # ('soccer_shuffled', '_tid_', None, False,
    #  None,
    #  None, True, 200000, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'all', False, False),
    #
    # ('chicago_num', None, ['Pickup Centroid Latitude', 'Pickup Centroid Longitude',
    #                        'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude',
    #                        'Fare', 'Tips', 'Tolls', 'Extras',
    #                        'Trip Total', 'Trip Seconds', 'Trip Miles'], True,
    #  [(100, ['Pickup Centroid Latitude', 'Pickup Centroid Longitude']),
    #   (100, ['Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']),
    #   (100, ['Fare']), (100, ['Tips']), (100, ['Tolls']), (100, ['Extras']),
    #   (100, ['Trip Total']), (100, ['Trip Seconds']), (100, ['Trip Miles'])],
    #  None, True, 400000, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'dk', False, False),
    #
    # ('chicago_num', None, ['Pickup Centroid Latitude', 'Pickup Centroid Longitude',
    #                        'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude',
    #                        'Fare', 'Tips', 'Tolls', 'Extras',
    #                        'Trip Total', 'Trip Seconds', 'Trip Miles'], True,
    #  [(100, ['Pickup Centroid Latitude', 'Pickup Centroid Longitude']),
    #   (100, ['Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']),
    #   (100, ['Fare']), (100, ['Tips']), (100, ['Tolls']), (100, ['Extras']),
    #   (100, ['Trip Total']), (100, ['Trip Seconds']), (100, ['Trip Miles'])],
    #  None, True, 400000, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'all', False, False),
    #
    # ('chicago_num_shuffled', '_tid_', ['Pickup Centroid Latitude', 'Pickup Centroid Longitude',
    #                                    'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude',
    #                                    'Fare', 'Tips', 'Tolls', 'Extras',
    #                                    'Trip Total', 'Trip Seconds', 'Trip Miles'], True,
    #  [(100, ['Pickup Centroid Latitude', 'Pickup Centroid Longitude']),
    #   (100, ['Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']),
    #   (100, ['Fare']), (100, ['Tips']), (100, ['Tolls']), (100, ['Extras']),
    #   (100, ['Trip Total']), (100, ['Trip Seconds']), (100, ['Trip Miles'])],
    #  None, True, 400000, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'dk', False, False),
    #
    # ('chicago_num_shuffled', '_tid_', ['Pickup Centroid Latitude', 'Pickup Centroid Longitude',
    #                                    'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude',
    #                                    'Fare', 'Tips', 'Tolls', 'Extras',
    #                                    'Trip Total', 'Trip Seconds', 'Trip Miles'], True,
    #  [(100, ['Pickup Centroid Latitude', 'Pickup Centroid Longitude']),
    #   (100, ['Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']),
    #   (100, ['Fare']), (100, ['Tips']), (100, ['Tolls']), (100, ['Extras']),
    #   (100, ['Trip Total']), (100, ['Trip Seconds']), (100, ['Trip Miles'])],
    #  None, True, 400000, 0.01, 'NaiveBayes',
    #  0.9, 100, 0.05, 0.3,
    #  'all', False, False),
]

approaches = ['C']
avg_time_iterations = None

for (dataset_name, entity_col, numerical_attrs, do_quantization,
     num_attr_groups_bins,
     tuples_to_read_list, model_monitoring, dataset_size, dataset_fraction_for_batch, estimator_type,
     weak_label_thresh, max_domain, cor_strength, nb_cor_strength,
     infer_mode, global_features, train_using_all_batches) in datasets:
    inc_args['dataset_name'] = dataset_name
    inc_args['entity_col'] = entity_col
    inc_args['numerical_attrs'] = numerical_attrs
    inc_args['do_quantization'] = do_quantization
    inc_args['num_attr_groups_bins'] = num_attr_groups_bins
    inc_args['tuples_to_read_list'] = tuples_to_read_list
    inc_args['model_monitoring'] = model_monitoring
    inc_args['dataset_size'] = dataset_size
    inc_args['dataset_fraction_for_batch'] = dataset_fraction_for_batch
    hc_args['estimator_type'] = estimator_type
    hc_args['weak_label_thresh'] = weak_label_thresh
    hc_args['max_domain'] = max_domain
    hc_args['cor_strength'] = cor_strength
    hc_args['nb_cor_strength'] = nb_cor_strength
    hc_args['infer_mode'] = infer_mode
    hc_args['global_features'] = global_features
    hc_args['train_using_all_batches'] = train_using_all_batches

    ######################################################################

    if 'A' in approaches:
        hc_args['incremental'] = False
        hc_args['repair_previous_errors'] = False
        hc_args['recompute_from_scratch'] = False
        hc_args['save_load_checkpoint'] = False
        hc_args['append'] = True
        inc_args['approach'] = 'co_a'

        # A - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [1]
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
        hc_args['save_load_checkpoint'] = False
        hc_args['append'] = False
        inc_args['approach'] = 'co_b'

        # B - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [1]
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

        if inc_args['model_monitoring']:
            hc_args['recompute_from_scratch'] = True
            hc_args['save_load_checkpoint'] = True
        else:
            hc_args['recompute_from_scratch'] = False
            hc_args['save_load_checkpoint'] = False

        hc_args['append'] = False
        inc_args['approach'] = 'co_c'

        # C - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [1]

        if inc_args['model_monitoring']:
            model_monitoring_input = build_model_monitoring_input(dataset_size, dataset_fraction_for_batch)

            for skip_training_starting_batch, tuples_to_read in model_monitoring_input.items():
                inc_args['skip_training_starting_batch'] = skip_training_starting_batch
                inc_args['tuples_to_read_list'] = tuples_to_read

                executor = Executor(hc_args, inc_args)
                executor.run()

                hc_args['skip_training'] = False
                hc_args['train_using_all_batches'] = train_using_all_batches
                hc_args['is_first_batch'] = True
        else:
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
        hc_args['append'] = False
        inc_args['approach'] = 'co_bplus'

        # B+ - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [1]
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
        hc_args['append'] = False
        inc_args['approach'] = 'co_cplus'

        # C+ - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [1]
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
        hc_args['append'] = False
        inc_args['approach'] = 'co_full'

        # Full - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['tuples_to_read_list'] = [sum(inc_args['tuples_to_read_list'])]
        inc_args['iterations'] = [1]
        executor = Executor(hc_args, inc_args)
        executor.run()

        if avg_time_iterations:
            # Full - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            executor = Executor(hc_args, inc_args)
            executor.run()
