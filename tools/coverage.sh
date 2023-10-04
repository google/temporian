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

# NOTE: if adding any flags to the `unittest discover` command, you must also
# define them in temporian/test/utils.py for abseil not to fail when parsing them.

# Flags explained:
# --omit: exclude files from coverage report
# --pattern: only run tests matching this pattern
# --verbose: print test names
# --buffer: dont print test output to stdout
# --failfast: fail on first test failure/error (else exits with non-error code even if tests fail)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="${PKDIR}/:$PYTHONPATH" coverage run --omit "*test*.py,*/test/*" -m unittest discover --pattern '*test*.py' --verbose --buffer --failfast
