#!/usr/bin/env bash

# Sets environment variables and moves to the root directory.
source ../set_env.sh

script="feature_generator.py"
if [ $# -eq 1 ] ; then
  script="$1"
fi

db_port=5432
dataset_name="nypd6"
entity_col="_"
dataset_size=32400
weak_label_thresh=0.9
cor_strength=0.05
nb_cor_strength=0.3
number_of_batches=100
number_of_executions=4

db_suffix=1
python $script $db_port $dataset_name $entity_col $dataset_size $weak_label_thresh $cor_strength $nb_cor_strength $number_of_batches $number_of_executions $db_suffix 2> output_"$dataset_name"_"$db_suffix".log &

db_suffix=2
python $script $db_port $dataset_name $entity_col $dataset_size $weak_label_thresh $cor_strength $nb_cor_strength $number_of_batches $number_of_executions $db_suffix 2> output_"$dataset_name"_"$db_suffix".log &

db_suffix=3
python $script $db_port $dataset_name $entity_col $dataset_size $weak_label_thresh $cor_strength $nb_cor_strength $number_of_batches $number_of_executions $db_suffix 2> output_"$dataset_name"_"$db_suffix".log &

db_suffix=4
python $script $db_port $dataset_name $entity_col $dataset_size $weak_label_thresh $cor_strength $nb_cor_strength $number_of_batches $number_of_executions $db_suffix 2> output_"$dataset_name"_"$db_suffix".log &

wait
