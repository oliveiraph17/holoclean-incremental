#!/usr/bin/env bash
source ../set_env.sh

script="baseline_2.py"
for i in {1..10}
do
    python $script 2>&1 >/dev/null | grep 'EXECUTION_TIME' >> ../experiments/hospital_10_batches/execution_time/baseline_2.txt
done

script="proposal_3.py"
for i in {1..10}
do
    python $script 2>&1 >/dev/null | grep 'EXECUTION_TIME' >> ../experiments/hospital_10_batches/execution_time/proposal_3.txt
done

script="proposal_4.py"
for i in {1..10}
do
    python $script 2>&1 >/dev/null | grep 'EXECUTION_TIME' >> ../experiments/hospital_10_batches/execution_time/proposal_4.txt
done

script="proposal_5.py"
for i in {1..10}
do
    python $script 2>&1 >/dev/null | grep 'EXECUTION_TIME' >> ../experiments/hospital_10_batches/execution_time/proposal_5.txt
done

script="proposal_6.py"
for i in {1..10}
do
    python $script 2>&1 >/dev/null | grep 'EXECUTION_TIME' >> ../experiments/hospital_10_batches/execution_time/proposal_6.txt
done

script="proposal_7.py"
for i in {1..10}
do
    python $script 2>&1 >/dev/null | grep 'EXECUTION_TIME' >> ../experiments/hospital_10_batches/execution_time/proposal_7.txt
done
