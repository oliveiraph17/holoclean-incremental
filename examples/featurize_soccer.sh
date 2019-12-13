#!/usr/bin/env bash

# Sets environment variables and moves to the root directory.
source ../set_env.sh

script="feature_generator.py"
# if [ $# -eq 1 ] ; then
#   script="$1"
# fi

db_port=$3
dataset_name="soccer"
entity_col="_"
dataset_size=200000
weak_label_thresh=0.9
cor_strength=0.05
nb_cor_strength=0.3
init_batch=$1
final_batch=$2
db_suffix=$init_batch
global=0

createdb -p $db_port -U holocleanuser -O holocleanuser -w -e holo_"$dataset_name"_"$db_suffix"
python $script $db_port $dataset_name $entity_col $batch_size $weak_label_thresh $cor_strength $nb_cor_strength $init_batch $final_batch $db_suffix $global 2> output_"$dataset_name"_"$db_suffix".log &

wait
