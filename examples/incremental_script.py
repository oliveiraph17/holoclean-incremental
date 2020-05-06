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
    'domain_thresh_2': 0,
    'weak_label_thresh': float(sys.argv[6]),
    'max_domain': int(sys.argv[7]),
    'cor_strength': float(sys.argv[8]),
    'nb_cor_strength': float(sys.argv[9]),
    'epochs': 20,
    'threads': 1,
    'verbose': False,
    'timeout': 3 * 60000,
    'estimator_type': 'NaiveBayes',
    'epochs_convergence': 0,
    'convergence_thresh': 0.01,
    'skip_training_thresh': 101,
    'log_repairing_quality': True,
    'log_execution_times': True,
    'log_feature_weights': True,
    'infer_mode': sys.argv[10],
    'group_models': str(sys.argv[11]) if str(sys.argv[11]) != '_' else None,
    'group_models_thresh': float(sys.argv[12]) if str(sys.argv[12]) != '_' else None,
    'skip_training_kl': str(sys.argv[13]) if str(sys.argv[13]) != '_' else None,
    'skip_training_kl_thresh': float(sys.argv[14]) if str(sys.argv[14]) != '_' else None,
}

approach = str(sys.argv[1])
avg_time_iterations = None
batch_size = int(sys.argv[4])
num_batches = int(sys.argv[5])

inc_args = {
    'project_root': os.environ['HOLOCLEANHOME'],
    'dataset_dir': os.environ['HOLOCLEANHOME'] + '/testdata/',
    'log_dir': os.environ['HOLOCLEANHOME'] + '/experimental_results/',
    'dataset_name': str(sys.argv[2]),
    'entity_col': str(sys.argv[3]) if str(sys.argv[3]) != '_' else None,
    'numerical_attrs': None,
    'tuples_to_read_list': [batch_size] * num_batches,
    'model_monitoring': False,
    'dataset_size': None,
    'iterations': [0],
    'do_quantization': False,
}


def get_approach_string():
    approach_string = 'co_' + approach.lower() + '_' + hc_args['infer_mode']
    if approach != 'Full':
        approach_string += '_' + str(num_batches) + 'batches'
    if hc_args['group_models']:
        approach_string += '_' + hc_args['group_models'] + str(hc_args['group_models_thresh']).replace('.', '')
    if hc_args['skip_training_kl']:
        approach_string += '_' + hc_args['skip_training_kl'] + str(hc_args['skip_training_kl_thresh']).replace('.', '')

    return approach_string


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

######################################################################


inc_args['approach'] = get_approach_string()

if approach == 'A':
    hc_args['append'] = True
elif approach == 'B':
    hc_args['incremental'] = True
    hc_args['repair_previous_errors'] = True
    hc_args['recompute_from_scratch'] = True
    hc_args['train_using_all_batches'] = True
elif approach == 'C':
    hc_args['incremental'] = True
    hc_args['train_using_all_batches'] = True
    hc_args['save_load_checkpoint'] = True
elif approach == 'B+':
    hc_args['incremental'] = True
    hc_args['repair_previous_errors'] = True
    hc_args['save_load_checkpoint'] = True
elif approach == 'C+':
    hc_args['incremental'] = True
    hc_args['save_load_checkpoint'] = True
else:
    # 'Full': uses only default parameters
    pass

# Quality
run_executor()

if avg_time_iterations:
    # Time
    hc_args['log_repairing_quality'] = False
    inc_args['iterations'] = avg_time_iterations
    run_executor()


# Sample of execution parameters
# C hospital _ 250 4 0.99 10000 0.6 0.8 dk pair_corr 0.95 weighted_kl 0.01

# ('food5k_shuffled', '_tid_', None, False,
#  None,
#  [1000] * 5, False, 5000, 0.2, 'NaiveBayes',
#  0.6, 1000, 0.2, 0.3,
#  'dk', False, False),

# ('hospital_numerical', None, ['Score', 'Sample'], True,
#  [(100, ['Score']), (150, ['Sample'])],
#  [250] * 4, False, 1000, 0.25, 'NaiveBayes',
#  0.99, 10000, 0.6, 0.8,
#  'dk', False, False),

# ('nypd6', None, None, False,
#  None,
#  [324] * 100, False, 32400, 0.01, 'NaiveBayes',
#  0.9, 10000, 0.05, 0.3,
#  'dk', False, None),

# ('soccer', None, None, False,
#  None,
#  [2000] * 100, False, 200000, 0.01, 'NaiveBayes',
#  0.9, 10000, 0.05, 0.3,
#  'dk', False, None),

# ('chicago_num_shuffled', '_tid_', ['Pickup Centroid Latitude', 'Pickup Centroid Longitude',
#                                    'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude',
#                                    'Fare', 'Tips', 'Tolls', 'Extras',
#                                    'Trip Total', 'Trip Seconds', 'Trip Miles'], True,
#  [(100, ['Pickup Centroid Latitude', 'Pickup Centroid Longitude']),
#   (100, ['Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']),
#   (100, ['Fare']), (100, ['Tips']), (100, ['Tolls']), (100, ['Extras']),
#   (100, ['Trip Total']), (100, ['Trip Seconds']), (100, ['Trip Miles'])],
#  [4000] * 100, False, 400000, 0.01, 'NaiveBayes',
#  0.9, 100, 0.05, 0.3,
#  'dk', False, None),
