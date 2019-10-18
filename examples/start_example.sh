#!/usr/bin/env bash
# Set & move to home directory
source ../set_env.sh

script="incremental_script.py"
if [ $# -eq 1 ] ; then
  script="$1"
fi

echo "Launching example script $script"
python $script

#scp finished.txt incuser@129.97.171.159:/home/incuser

