#!/usr/bin/env bash
source ../set_env.sh

script="baseline_2.py"
python $script 2>&1 >/dev/null | grep 'EXPERIMENT' >> ../experiments/hospital_2_batches/baseline_2.txt

script="proposal_3.py"
python $script 2>&1 >/dev/null | grep 'EXPERIMENT' >> ../experiments/hospital_2_batches/proposal_3.txt

script="proposal_4.py"
python $script 2>&1 >/dev/null | grep 'EXPERIMENT' >> ../experiments/hospital_2_batches/proposal_4.txt

script="proposal_5.py"
python $script 2>&1 >/dev/null | grep 'EXPERIMENT' >> ../experiments/hospital_2_batches/proposal_5.txt

script="proposal_6.py"
python $script 2>&1 >/dev/null | grep 'EXPERIMENT' >> ../experiments/hospital_2_batches/proposal_6.txt

script="proposal_7.py"
python $script 2>&1 >/dev/null | grep 'EXPERIMENT' >> ../experiments/hospital_2_batches/proposal_7.txt

