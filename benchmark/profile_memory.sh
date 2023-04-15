#!/bin/bash


# print commands as they're executed, exit on error
set -ve

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# fail if script name isn't passed as first positional arg
if [[ $# -eq 0 ]] ; then
    echo 'Pass name of script inside benchmark/scripts as first positional arg.'
    exit 1
fi

# read script name
script_name=$1
shift

# pass -p flag to get plot instead of line-by-line
if getopts "p:" arg; then
    # display memory over time plot
    mprof run --include-children --python python benchmark/scripts/$script_name.py
    mprof plot --flame
else
    # display line-by-line memory consumption
    python -m memory_profiler --include-children benchmark/scripts/$script_name.py
fi