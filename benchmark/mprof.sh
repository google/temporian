export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# TODO: provide flag to `bazel run benchmark:memory` to choose between
# line-by-line memory consumption and memory over time plot

# display line-by-line memory consumption
python -m memory_profiler benchmark/basic.py

# display memory over time plot
# mprof run --python python benchmark/basic.py
# mprof plot --flame