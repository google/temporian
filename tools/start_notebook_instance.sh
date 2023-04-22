#!/bin/bash

# Compile Temporian and start a jupyter notebook instance with it.
#
# Usage example:
#   ./tools/start_notebook_instance.sh
#
#   In a notebook, you now use "import temporian as tp".
#   Make sure to re-run this command each time the source code of temporian is
#   changed.
#
#   Check example/notebook.ipynb for an example.
#

set -xev

# Build temporian
# ===============

bazel build -c opt //temporian

# Assemble files
# ==============

PKDIR="$(pwd)/build_package"
rm -fr ${PKDIR}
mkdir -p ${PKDIR}

# Copy python files
find temporian -name "*.py" -type f -exec cp --parents {} ${PKDIR}/ \;
# Copy compiled files
( cd bazel-bin && find temporian \( -name "*.so" -o -name "*.py" \) -type f -exec cp --parents {} ${PKDIR}/ \; )

tree ${PKDIR}

# Start notebook
# ==============

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="${PKDIR}/:$PYTHONPATH" jupyter notebook
