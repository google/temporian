export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python -m memory_profiler benchmark/basic.py

# display memory over time plot instead of line-by-line consumption:
# mprof run --python python benchmark/basic.py
# mprof plot --flame