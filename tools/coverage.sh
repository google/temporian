#!/bin/bash

# Compiles Temporian and runs test coverage (Python only)

set -vex

# Build temporian
bazel build -c opt //temporian

# Assemble files
PKDIR="$(pwd)/build_package"
rm -fr ${PKDIR}
mkdir -p ${PKDIR}

rsync -r --safe-links --include='*/' --include='*.py' --exclude='*' "temporian/" "${PKDIR}/temporian/"
rsync -r --safe-links --exclude='*/*.runfiles/' --include='*/' --include='*.py' --include='*.so' --exclude='*' "bazel-bin/temporian/" "${PKDIR}/temporian/"

# Run coverage
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="${PKDIR}/:$PYTHONPATH" coverage run --omit "*test.py,*/test/*" -m unittest discover --pattern '*test.py'
# Print coverage results
coverage report -m
