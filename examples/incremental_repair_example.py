import sys
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *
sys.path.append('../')

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
    verbose=True,
    timeout=3*60000,
    feature_norm=False,
    weight_norm=False,
    print_fw=True
).session

# Load existing data and Denial Constraints.
hc.load_data('hospital', '../testdata/hospital_0900.csv')
hc.load_dcs('../testdata/hospital_constraints.txt')
hc.ds.set_constraints(hc.get_dcs())

# Detect erroneous cells using these two detectors.
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)

# Repair errors based on the defined features.
hc.setup_domain()
# featurizers = [
#     InitAttrFeaturizer(),
#     OccurAttrFeaturizer(),
#     FreqFeaturizer(),
#     ConstraintFeaturizer(),
# ]
# hc.repair_errors(featurizers)

# Evaluate the correctness of the results.
# hc.evaluate(fpath='../testdata/hospital_clean.csv',
#             tid_col='tid',
#             attr_col='attribute',
#             val_col='correct_val')

############################################################

batch = 2

# Load incoming data.
hc.load_new_data('hospital', '../testdata/hospital_0100.csv', batch)

hc.detect_errors(detectors, batch)

hc.setup_domain(batch)

############################################################

batch = 3

# Load incoming data.
# hc.load_new_data('hospital', '../testdata/hospital_0050_2.csv', batch)

# hc.detect_errors(detectors, batch)
