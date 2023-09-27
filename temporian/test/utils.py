import os
import sys

from absl import flags

# Define flags used when calling unittest in tools/coverage.sh, else fails when
# parsing them
flags.DEFINE_string("pattern", None, "")

# Parse flags, else fails when accessing FLAGS.test_srcdir when running tests
# with unittest directly
flags.FLAGS(sys.argv)


def get_test_data_path(path: str) -> str:
    """Returns the path to a test data file relative to the project's root, e.g.
    temporian/test/test_data/io/input.csv.

    Necessary when accessing these files in Bazel-ran tests."""
    dir = flags.FLAGS.test_srcdir

    # If test_srcdir is not set, we are not running in Bazel, return the path.
    if dir == "":
        return path

    return os.path.join(flags.FLAGS.test_srcdir, "temporian", path)
