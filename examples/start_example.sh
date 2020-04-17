#!/usr/bin/env bash
# Set & move to home directory
source ../set_env.sh

script="incremental_script.py"
if [ $# -eq 1 ] ; then
  script="$1"
fi

approach="Full"
dataset_name="hospital_shuffled"
entity_col="_tid_"
batch_size=1000
number_of_batches=1
weak_label_thresh="0.99"
max_domain=1000000
cor_strength="0.6"
nb_cor_strength="0.8"

echo "Launching script $script"
python "$script" $approach $dataset_name $entity_col $batch_size $number_of_batches \
       $weak_label_thresh $max_domain $cor_strength $nb_cor_strength

approach="A"
batch_size=10
number_of_batches=100

echo "Launching script $script"
python "$script" $approach $dataset_name $entity_col $batch_size $number_of_batches \
       $weak_label_thresh $max_domain $cor_strength $nb_cor_strength

#('Full', 'food5k_shuffled', '_tid_', [5000], 0.6, 500, 0.2, 0.3)
#('A', 'food5k_shuffled', '_tid_', [50] * 100, 0.6, 500, 0.2, 0.3)
#('Full', 'nypd6', None, [32399], 0.9, 60, 0.05, 0.3)
#('A', 'nypd6', None, [324] * 100, 0.9, 60, 0.05, 0.3)
#('Full', 'soccer', None, [200000], 0.9, 40, 0.05, 0.3)
#('A', 'soccer', None, [2000] * 100, 0.9, 40, 0.05, 0.3)
