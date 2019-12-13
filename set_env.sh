#!/usr/bin/env bash

export HOLOCLEANHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Holoclean home directory: $HOLOCLEANHOME"
export PYTHONPATH="$PYTHONPATH:$HOLOCLEANHOME"
export PATH="$PATH:$HOLOCLEANHOME"
echo $PATH
export PGPASSWORD="abcd1234"
echo "Environment variables set!"
