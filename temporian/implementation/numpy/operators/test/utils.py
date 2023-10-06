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
import time
from absl.testing import absltest

from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.core.operators.base import Operator
from temporian.core import serialization


def assertEqualEventSet(
    test: absltest.TestCase, real: EventSet, expected: EventSet
):
    """Asserts the equality between real and expected.

    Prints a nice message in case of error.
    """

    test.assertEqual(
        real,
        expected,
        (
            "\n==========\nREAL:\n==========\n"
            f"{real}"
            "\n==========\nEXPECTED:\n==========\n"
            f"{expected}"
        ),
    )


def assertEqualDFRandomRowOrder(
    test: absltest.TestCase, real: "pd.DataFrame", expected: "pd.DataFrame"
):
    row_real = set([str(row.to_dict()) for _, row in real.iterrows()])
    row_expected = set([str(row.to_dict()) for _, row in expected.iterrows()])
    test.assertEqual(
        row_real,
        row_expected,
        (
            "\n==========\nREAL:\n==========\n"
            f"{real}"
            "\n==========\nEXPECTED:\n==========\n"
            f"{expected}"
        ),
    )


def testOperatorAndImp(
    test: absltest.TestCase, op: Operator, imp: OperatorImplementation
):
    """Tests an operator and its implementation.

    Currently test:
      - Serialization / unserialization of the operator.
    """

    # TODO: Add tests related to the implementation.
    del imp

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
