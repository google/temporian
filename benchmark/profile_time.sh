#!/bin/bash


# print commands as they're executed, exit on error
set -ve

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# fail if script name isn't passed as first positional arg
if [[ $# -eq 0 ]] ; then
    echo 'Pass name of script inside benchmark/scripts as first positional arg.'
    exit 1
fi

scalene --reduced-profile benchmark/scripts/$1.py