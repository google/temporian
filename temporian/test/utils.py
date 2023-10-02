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


import numpy as np
from absl.testing import absltest

from temporian.core import serialization
from temporian.implementation.numpy.data.event_set import EventSet


def _f64(l):
    return np.array(l, np.float64)


def _f32(l):
    return np.array(l, np.float32)


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
    # Result is the expected one

    test.assertEqual(result, expected)
    result.check_same_sampling(expected)

    # Graph can be serialized and deserialized

    if result.creator is None:
        raise ValueError("EventSet has no creator.")
    op = result.creator

    serialized_op = serialization._serialize_operator(op)

    nodes = {}
    for node in op.inputs.values():
        nodes[serialization._identifier(node)] = node
    for node in op.outputs.values():
        nodes[serialization._identifier(node)] = node

    _ = serialization._unserialize_operator(serialized_op, nodes)
