#!/bin/bash

# Compile Temporian and execute all notebook cells
#
# Usage example:
#   ./tools/run_notebooks.sh docs/src/user_guide.ipynb
#

set -vex

# Build temporian
bazel build -c opt --config=linux //temporian

# Assemble files
PKDIR="$(pwd)/build_package"
rm -fr ${PKDIR}
mkdir -p ${PKDIR}

rsync -r --safe-links --exclude='*/*test/' --include='*/' --include='*.py' --exclude='*' "temporian/" "${PKDIR}/temporian/"
rsync -r --safe-links --exclude='*/*.runfiles/' --include='*/' --include='*.py' --include='*.so' --exclude='*' "bazel-bin/temporian/" "${PKDIR}/temporian/"

# Run the notebooks and overwrites them with the outputs
for path in "$@"
do
    time PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="${PKDIR}/:$PYTHONPATH" jupyter nbconvert --debug --ExecutePreprocessor.timeout=600 --execute $path --to notebook --inplace
done
