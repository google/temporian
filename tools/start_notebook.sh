#!/bin/bash

# Compile Temporian and start a jupyter notebook instance with it.
#
# Usage example:
#   ./tools/start_notebook.sh
#   # Then open the url, or set the url in code studio Jupyter server.
#
# In a notebook, you now use "import temporian as tp".
# Make sure to re-run this command each time the source code of temporian is
# changed.
#
# Check example/toy_example.ipynb for an example.

set -vex

# Build temporian
bazel build -c opt  --config=linux //temporian

# Assemble files
PKDIR="$(pwd)/build_package"
rm -fr ${PKDIR}
mkdir -p ${PKDIR}

rsync -v -r --safe-links --exclude='*/*test/' --include='*/' --include='*.py' --exclude='*' "temporian/" "${PKDIR}/temporian/"
rsync -v -r --safe-links --exclude='*/*.runfiles/' --include='*/' --include='*.py' --include='*.so' --exclude='*' "bazel-bin/temporian/" "${PKDIR}/temporian/"

# Start notebook
# Note: Use "notebook" or "lab" ("jupyterlab").
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="${PKDIR}/:$PYTHONPATH" jupyter-lab --no-browser
