#!/bin/bash

# Builds all temporian for all supported python versions.
# See tools/build_manylinux.sh for more details.

set -x
set -e

for PYTHON_VERSION in 38 39 310 311; do
    PYTHON_VERSION=$PYTHON_VERSION tools/build_manylinux.sh
done