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
# Check examples/tutorials/getting_started.ipynb for an example.

set -vex

# Build temporian
bazel build -c opt //temporian

# Assemble files
PKDIR="$(pwd)/build_package"
rm -fr ${PKDIR}
mkdir -p ${PKDIR}

rsync -v -r --include='*/' --include='*.py' --exclude='*' "temporian/" "${PKDIR}/temporian/"
rsync -v -r --include='*/' --include='*.py' --include='*.so' --exclude='*' --exclude='test' "bazel-bin/temporian/" "${PKDIR}/temporian/"

if [ $1 == "test" ]; then
    # Checks that the code in all notebooks run without errors
    for path in $(ls examples/tutorials/*.ipynb); do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="${PKDIR}/:$PYTHONPATH" jupyter nbconvert --execute $path --to python --stdout
    done
else
    # Start notebook
    # Note: Use "notebook" or "lab" ("jupyterlab").
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="${PKDIR}/:$PYTHONPATH" jupyter-lab --no-browser
fi
