#!/usr/bin/env bash
source ../set_env.sh

################################################################################
# Repairing quality
################################################################################

dir_path="adult_1/one_shot"

script_file="baseline_1.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

dir_path="adult_1/4_batches"

script_file="baseline_2.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="proposal_3.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="proposal_4.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="proposal_5.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="proposal_6.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="proposal_7.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

################################################################################
# Execution times
################################################################################

dir_path="adult_1/one_shot"

script_file="baseline_1_times.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

dir_path="adult_1/4_batches"

script_file="baseline_2_times.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="proposal_3_times.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="proposal_4_times.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="baseline_5_times.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="proposal_6_times.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null

script_file="proposal_7_times.py"
script_path="$dir_path/$script_file"
python ${script_path} 2>/dev/null
