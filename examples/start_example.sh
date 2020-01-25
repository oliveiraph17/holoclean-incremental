#!/usr/bin/env bash
# Set & move to home directory
source ../set_env.sh

script="$1"

echo "Launching example script $script"
python $script
# python $script $2 $3 $4 $5

#scp finished.txt incuser@129.97.171.159:/home/incuser

