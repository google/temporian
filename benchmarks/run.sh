set -vex
ls
pwd

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
scalene benchmarks/complete.py

#bazel-bin/temporian/implementation/numpy_cc/operators/
#DYLD_INSERT_LIBRARIES=/mnt/g/projects/work/temporian/bazel-bin/temporian/implementation/numpy_cc/operators
#cd bazel-bin & python benchmarks/complete.py