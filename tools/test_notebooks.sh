#!/bin/bash

# Execute all notebook files under examples/tutorials to check that they work
#
# Usage example:
#   ./tools/test_notebooks.sh

# Checks that the code in all notebooks run
for path in $(ls examples/tutorials/*.ipynb); do
    poetry run jupyter nbconvert --execute $path --to python --stdout
done
