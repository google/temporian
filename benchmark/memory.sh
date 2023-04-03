#!/bin/bash

# print commands as they're executed, exit on error
set -ve

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# TODO: provide flag to `bazel run benchmark:memory` to choose between
# line-by-line memory consumption and memory over time plot

if getopts "p:" arg; then
    # display memory over time plot
    mprof run --python python benchmark/scripts/basic.py
    mprof plot --flame
else
    # display line-by-line memory consumption
    python -m memory_profiler benchmark/scripts/basic.py
fi