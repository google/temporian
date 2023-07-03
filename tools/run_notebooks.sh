#!/bin/bash

# Compile Temporian and execute all notebook cells
#
# Usage example:
#   ./tools/test_notebooks.sh
#

set -vex

# Build temporian
bazel build -c opt //temporian

# Assemble files
PKDIR="$(pwd)/build_package"
rm -fr ${PKDIR}
mkdir -p ${PKDIR}

rsync -r --safe-links --exclude='*/*test/' --include='*/' --include='*.py' --exclude='*' "temporian/" "${PKDIR}/temporian/"
rsync -r --safe-links --exclude='*/*.runfiles/' --include='*/' --include='*.py' --include='*.so' --exclude='*' "bazel-bin/temporian/" "${PKDIR}/temporian/"

# Checks that the code in all notebooks run without errors
for path in $(ls examples/tutorials/*.ipynb); do
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="${PKDIR}/:$PYTHONPATH" jupyter nbconvert --execute $path --to notebook --inplace
done
