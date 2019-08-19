#!/usr/bin/env bash
source ../set_env.sh

script="baseline_1.py"
python $script 2>/dev/null

script="baseline_1_times.py"
python $script 2>/dev/null

