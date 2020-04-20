from examples.holoclean_incremental_repair_example import Executor

import os
import resource
import sys

from time import sleep
from concurrent.futures import ThreadPoolExecutor


class MemoryMonitor:
    def __init__(self):
        self.keep_measuring = True

    def measure_usage(self):
        max_usage = 0
        while self.keep_measuring:
            # Keeps track of the maximum resident set size (in KB since Linux 2.6.26).
            max_usage = max(
                max_usage,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss +
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            )
            sleep(0.01)

        return max_usage


hc_args = {
    'detectors': [('nulldetector', 'NullDetector', False),
                  ('violationdetector', 'ViolationDetector', False)],
    # 'detectors': [('errorloaderdetector', 'ErrorsLoaderDetector', True)],
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
    'skip_training_thresh': 101,
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
    'append': True,
    'infer_mode': 'dk',
    'train_using_all_batches': None,
}

inc_args = {
    'project_root': os.environ['HOLOCLEANHOME'],
    'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
    'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
    'dataset_name': None,
    'entity_col': None,
    'numerical_attrs': None,
    'approach': 'co_a',
    'tuples_to_read_list': [250] * 4,
    'iterations': [0],
}

params = {
    'approach': str(sys.argv[1]),
    'dataset_name': str(sys.argv[2]),
    'entity_col': str(sys.argv[3]) if str(sys.argv[3]) != '_' else None,
    'tuples_to_read_list': [int(sys.argv[4])] * int(sys.argv[5]) if int(sys.argv[5]) != 1 else [int(sys.argv[4])],
    'weak_label_thresh': float(sys.argv[6]),
    'max_domain': int(sys.argv[7]),
    'cor_strenth': float(sys.argv[8]),
    'nb_cor_strength': float(sys.argv[9])
}

datasets = [(params['approach'], params['dataset_name'], params['entity_col'], params['tuples_to_read_list'],
             params['weak_label_thresh'], params['max_domain'], params['cor_strenth'], params['nb_cor_strength'])]

avg_time_iterations = None


def run_executor():
    with ThreadPoolExecutor() as thread_pool_executor:
        memory_monitor = MemoryMonitor()
        memory_monitor_thread = thread_pool_executor.submit(memory_monitor.measure_usage)
        try:
            executor = Executor(hc_args, inc_args)

            executor_thread = thread_pool_executor.submit(executor.run)
            executor_thread.result()
        finally:
            memory_monitor.keep_measuring = False

            max_usage_kb = memory_monitor_thread.result()
            # Writes to file the maximum resident set size in GB.
            max_usage_gb = (max_usage_kb / 1024) / 1024
            with open(inc_args['log_dir'] + inc_args['dataset_name'] + '/' +
                      inc_args['approach'] + '_memory_log.csv', 'a') as f:
                f.writelines([str(max_usage_gb) + ' GB\n'])


for (approach, dataset_name, entity_col, tuples_to_read_list,
     weak_label_thresh, max_domain, cor_strength, nb_cor_strength) in datasets:
    inc_args['dataset_name'] = dataset_name
    inc_args['entity_col'] = entity_col
    inc_args['tuples_to_read_list'] = tuples_to_read_list
    hc_args['weak_label_thresh'] = weak_label_thresh
    hc_args['max_domain'] = max_domain
    hc_args['cor_strength'] = cor_strength
    hc_args['nb_cor_strength'] = nb_cor_strength

    ######################################################################

    if approach == 'A':
        hc_args['incremental'] = False
        hc_args['repair_previous_errors'] = False
        hc_args['recompute_from_scratch'] = False
        hc_args['train_using_all_batches'] = False
        inc_args['approach'] = 'co_a'

        # A - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [0]
        run_executor()

        if avg_time_iterations:
            # A - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            run_executor()

    ######################################################################

    if approach == 'B':
        hc_args['incremental'] = True
        hc_args['repair_previous_errors'] = True
        hc_args['recompute_from_scratch'] = True
        hc_args['train_using_all_batches'] = True
        inc_args['approach'] = 'co_b'

        # B - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [0]
        run_executor()

        if avg_time_iterations:
            # B - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            run_executor()

    ######################################################################

    if approach == 'C':
        hc_args['incremental'] = True
        hc_args['repair_previous_errors'] = False
        hc_args['recompute_from_scratch'] = False
        hc_args['train_using_all_batches'] = True
        inc_args['approach'] = 'co_c'

        # C - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [0]
        run_executor()

        if avg_time_iterations:
            # C - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            run_executor()

    ######################################################################

    if approach == 'B+':
        hc_args['incremental'] = True
        hc_args['repair_previous_errors'] = True
        hc_args['recompute_from_scratch'] = True
        hc_args['save_load_checkpoint'] = True
        inc_args['approach'] = 'co_bplus'

        # B+ - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [0]
        run_executor()

        if avg_time_iterations:
            # B+ - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            run_executor()

    ######################################################################

    if approach == 'C+':
        hc_args['incremental'] = True
        hc_args['repair_previous_errors'] = False
        hc_args['recompute_from_scratch'] = False
        hc_args['save_load_checkpoint'] = True
        inc_args['approach'] = 'co_cplus'

        # C+ - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [0]
        run_executor()

        if avg_time_iterations:
            # C+ - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            run_executor()

    ######################################################################

    if approach == 'Full':
        hc_args['incremental'] = False
        hc_args['repair_previous_errors'] = False
        hc_args['recompute_from_scratch'] = False
        hc_args['save_load_checkpoint'] = False
        inc_args['approach'] = 'co_full'

        # Full - Quality
        hc_args['log_repairing_quality'] = True
        hc_args['log_execution_times'] = True
        inc_args['iterations'] = [0]
        run_executor()

        if avg_time_iterations:
            # Full - Time
            hc_args['log_repairing_quality'] = False
            hc_args['log_execution_times'] = True
            inc_args['iterations'] = avg_time_iterations
            run_executor()
