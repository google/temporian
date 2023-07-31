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

import math

from absl.testing import absltest
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from temporian.core.operators.window.moving_count import MovingCountOperator
from temporian.implementation.numpy.operators.window.moving_count import (
    MovingCountNumpyImplementation,
    operators_cc,
)
from temporian.core.data import node as node_lib
from numpy.testing import assert_array_equal
from temporian.io.pandas import from_pandas


def _f64(l):
    return np.array(l, np.float64)


def _f32(l):
    return np.array(l, np.float32)


def _i32(l):
    return np.array(l, np.int32)


nan = math.nan


class MovingCountOperatorTest(absltest.TestCase):
    def test_cc_wo_sampling(self):
        assert_array_equal(
            operators_cc.moving_count(
                _f64([1, 2, 3, 5, 20]),
                5.0,
            ),
            _i32([1, 2, 3, 4, 1]),
        )

    def test_flat(self):
        """A simple event set."""

        evset = from_pandas(
            pd.DataFrame(
                [
                    [10.0, 20.0, 1],
                    [00.0, 21.0, 2],
                    [12.0, 00.0, 3],
                    [13.0, 23.0, 5],
                    [14.0, 24.0, 20],
                ],
                columns=["a", "b", "timestamp"],
            )
        )

        op = MovingCountOperator(
            input=evset.node(), window_length=5, sampling=None
        )
        op.outputs["output"].check_same_sampling(evset.node())

        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = MovingCountNumpyImplementation(op)

        output = instance(input=evset)

        expected_output = from_pandas(
            pd.DataFrame(
                [
                    [1, 1],
                    [2, 2],
                    [3, 3],
                    [4, 5],
                    [1, 20],
                ],
                columns=["count", "timestamp"],
            ).astype({"count": np.int32})
        )

        self.assertEqual(repr(output), repr({"output": expected_output}))

    def test_with_index(self):
        """Indexed Event sets."""

        evset = from_pandas(
            pd.DataFrame(
                [
                    ["X1", "Y1", 1],
                    ["X1", "Y1", 2],
                    ["X1", "Y1", 3],
                    ["X2", "Y1", 1.1],
                    ["X2", "Y1", 2.1],
                    ["X2", "Y1", 3.1],
                    ["X2", "Y2", 1.2],
                    ["X2", "Y2", 2.2],
                    ["X2", "Y2", 3.2],
                ],
                columns=["x", "y", "timestamp"],
            ),
            indexes=["x", "y"],
        )

        op = MovingCountOperator(
            input=evset.node(),
            window_length=5,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = MovingCountNumpyImplementation(op)

        output = instance(input=evset)

        expected_output = from_pandas(
            pd.DataFrame(
                [
                    ["X1", "Y1", 1, 1],
                    ["X1", "Y1", 2, 2],
                    ["X1", "Y1", 3, 3],
                    ["X2", "Y1", 1, 1.1],
                    ["X2", "Y1", 2, 2.1],
                    ["X2", "Y1", 3, 3.1],
                    ["X2", "Y2", 1, 1.2],
                    ["X2", "Y2", 2, 2.2],
                    ["X2", "Y2", 3, 3.2],
                ],
                columns=["x", "y", "count", "timestamp"],
            ).astype({"count": np.int32}),
            indexes=["x", "y"],
        )

        self.assertEqual(output["output"], expected_output)

    def test_with_sampling(self):
        """Event sets with user provided sampling."""

        evset = from_pandas(
            pd.DataFrame(
                [
                    [1],
                    [2],
                    [3],
                    [5],
                    [6],
                ],
                columns=["timestamp"],
            )
        )

        op = MovingCountOperator(
            input=evset.node(),
            window_length=3.1,
            sampling=node_lib.input_node([]),
        )
        self.assertEqual(
            op.list_matching_io_samplings(), [("sampling", "output")]
        )
        instance = MovingCountNumpyImplementation(op)

        sampling_data = from_pandas(
            pd.DataFrame(
                [
                    [-1.0],
                    [1.0],
                    [1.1],
                    [3.0],
                    [3.5],
                    [6.0],
                    [10.0],
                ],
                columns=["timestamp"],
            )
        )

        output = instance(input=evset, sampling=sampling_data)

        expected_output = from_pandas(
            pd.DataFrame(
                [
                    [0, -1.0],
                    [1, 1.0],
                    [1, 1.1],
                    [3, 3.0],
                    [3, 3.5],
                    [3, 6.0],
                    [0, 10.0],
                ],
                columns=["count", "timestamp"],
            ).astype({"count": np.int32})
        )

        self.assertEqual(output["output"], expected_output)


if __name__ == "__main__":
    absltest.main()
