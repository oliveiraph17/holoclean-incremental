#!/usr/bin/env bash

export HOLOCLEANHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "HoloClean home directory: $HOLOCLEANHOME"
export PYTHONPATH="$PYTHONPATH:$HOLOCLEANHOME"
export PATH="$PATH:$HOLOCLEANHOME"
echo $PATH
echo "Environment variables set!"
