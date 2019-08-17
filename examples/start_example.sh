#!/usr/bin/env bash
source ../set_env.sh

script="2_baseline_2.py"
python $script 2>/dev/null

script="2_proposal_3.py"
python $script 2>/dev/null

script="2_proposal_4.py"
python $script 2>/dev/null

script="2_proposal_5.py"
python $script 2>/dev/null

script="2_proposal_6.py"
python $script 2>/dev/null

script="2_proposal_7.py"
python $script 2>/dev/null

script="10_baseline_2.py"
python $script 2>/dev/null

script="10_proposal_3.py"
python $script 2>/dev/null

script="10_proposal_4.py"
python $script 2>/dev/null

script="10_proposal_5.py"
python $script 2>/dev/null

script="10_proposal_6.py"
python $script 2>/dev/null

script="10_proposal_7.py"
python $script 2>/dev/null
