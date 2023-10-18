# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time

import numpy as np
from absl import flags
from absl.testing import absltest

from temporian.core import serialization
from temporian.implementation.numpy.data.event_set import EventSet

# Define flags used when calling unittest in tools/coverage.sh, else fails when
# parsing them
flags.DEFINE_string("pattern", None, "")
flags.DEFINE_bool("verbose", None, "")
flags.DEFINE_bool("buffer", None, "")
flags.DEFINE_bool("failfast", None, "")

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


def f64(l):
    return np.array(l, np.float64)


def f32(l):
    return np.array(l, np.float32)


def i32(l):
    return np.array(l, np.int32)


def assertOperatorResult(
    test: absltest.TestCase, result: EventSet, expected: EventSet
):
    """Tests that the output of an operator is the expected one, and performs
    tests that are common to all operators.

    Extend with more checks as needed.

    Currently tests:
      - The result is the same as the expected output.
      - The result has the same sampling as the expected output.
      - Serialization / unserialization of the graph.
    """
    test.assertEqual(
        result,
        expected,
        (
            "\n==========\nRESULT:\n==========\n"
            f"{result}"
            "\n==========\nEXPECTED:\n==========\n"
            f"{expected}"
        ),
    )
    result.check_same_sampling(expected)

    # Graph can be serialized and deserialized

    if result.creator is None:
        raise ValueError("EventSet has no creator.")
    op = result.creator

    if op.definition.is_serializable:
        serialized_op = serialization._serialize_operator(op)
        nodes = {}
        for node in op.inputs.values():
            nodes[serialization._identifier(node)] = node
        for node in op.outputs.values():
            nodes[serialization._identifier(node)] = node

        _ = serialization._unserialize_operator(serialized_op, nodes)


class SetTimezone:
    def __init__(self, timezone: str = "America/Montevideo"):
        self._tz = timezone
        self._restore_tz = ""

    def __enter__(self):
        self._restore_tz = os.environ.get("TZ", "")
        os.environ["TZ"] = self._tz
        time.tzset()

    def __exit__(self, *args):
        os.environ["TZ"] = self._restore_tz
        time.tzset()
