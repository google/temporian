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
from numpy.testing import assert_almost_equal
import pandas as pd

from temporian.core.operators.window.moving_standard_deviation import (
    MovingStandardDeviationOperator,
)
from temporian.implementation.numpy.operators.window.moving_standard_deviation import (
    MovingStandardDeviationNumpyImplementation,
    operators_cc,
)
from temporian.core.data import node as node_lib
import math
from numpy.testing import assert_almost_equal
from temporian.implementation.numpy.data.io import pd_dataframe_to_event_set


def _f64(l):
    return np.array(l, np.float64)


def _f32(l):
    return np.array(l, np.float32)


nan = math.nan


class MovingStandardDeviationOperatorTest(absltest.TestCase):
    def test_cc_wo_sampling(self):
        assert_almost_equal(
            operators_cc.moving_standard_deviation(
                _f64([1, 2, 3, 5, 20]),
                _f32([10, nan, 12, 13, 14]),
                5.0,
            ),
            _f32([0.0, 0.0, 1.0, 1.247219, 0.0]),
        )

    def test_flat(self):
        """A simple event set."""

        input_data = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [10.0, 20.0, 1],
                    [11.0, 21.0, 2],
                    [12.0, 22.0, 3],
                    [13.0, 23.0, 5],
                    [14.0, 24.0, 20],
                ],
                columns=["a", "b", "timestamp"],
            )
        )

        op = MovingStandardDeviationOperator(
            input=input_data.node(),
            window_length=5.0,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = MovingStandardDeviationNumpyImplementation(op)

        output = instance(input=input_data)

        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 0, 1],
                    [0.5, 0.5, 2],
                    [math.sqrt(2 / 3), math.sqrt(2 / 3), 3],
                    [math.sqrt(1.25), math.sqrt(1.25), 5],
                    [0, 0, 20],
                ],
                columns=["a", "b", "timestamp"],
            )
        )

        self.assertEqual(repr(output), repr({"output": expected_output}))

    def test_with_index(self):
        """Indexed event sets."""

        input_data = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    ["X1", "Y1", 10.0, 1],
                    ["X1", "Y1", 11.0, 2],
                    ["X1", "Y1", 12.0, 3],
                    ["X2", "Y1", 13.0, 1.1],
                    ["X2", "Y1", 14.0, 2.1],
                    ["X2", "Y1", 15.0, 3.1],
                    ["X2", "Y2", 16.0, 1.2],
                    ["X2", "Y2", 17.0, 2.2],
                    ["X2", "Y2", 18.0, 3.2],
                ],
                columns=["x", "y", "a", "timestamp"],
            ),
            index_names=["x", "y"],
        )

        op = MovingStandardDeviationOperator(
            input=input_data.node(),
            window_length=5.0,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = MovingStandardDeviationNumpyImplementation(op)

        output = instance(input=input_data)

        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    ["X1", "Y1", 0, 1],
                    ["X1", "Y1", 0.5, 2],
                    ["X1", "Y1", math.sqrt(2 / 3), 3],
                    ["X2", "Y1", 0, 1.1],
                    ["X2", "Y1", 0.5, 2.1],
                    ["X2", "Y1", math.sqrt(2 / 3), 3.1],
                    ["X2", "Y2", 0, 1.2],
                    ["X2", "Y2", 0.5, 2.2],
                    ["X2", "Y2", math.sqrt(2 / 3), 3.2],
                ],
                columns=["x", "y", "a", "timestamp"],
            ),
            index_names=["x", "y"],
        )

        self.assertEqual(repr(output), repr({"output": expected_output}))

    def test_with_sampling(self):
        """Event sets with user provided sampling."""

        input_data = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [10.0, 1],
                    [11.0, 2],
                    [12.0, 3],
                    [13.0, 5],
                    [14.0, 6],
                ],
                columns=["a", "timestamp"],
            )
        )

        op = MovingStandardDeviationOperator(
            input=input_data.node(),
            window_length=3.1,
            sampling=node_lib.input_node([]),
        )
        self.assertEqual(
            op.list_matching_io_samplings(), [("sampling", "output")]
        )
        instance = MovingStandardDeviationNumpyImplementation(op)

        sampling_data = pd_dataframe_to_event_set(
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

        output = instance(input=input_data, sampling=sampling_data)

        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [math.nan, -1.0],
                    [0, 1.0],
                    [0, 1.1],
                    [math.sqrt(2 / 3), 3.0],
                    [math.sqrt(2 / 3), 3.5],
                    [math.sqrt(2 / 3), 6.0],
                    [math.nan, 10.0],
                ],
                columns=["a", "timestamp"],
            )
        )

        self.assertEqual(repr(output), repr({"output": expected_output}))

    def test_with_nan(self):
        """The input features contains nan values."""

        input_data = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [math.nan, 1],
                    [11.0, 2],
                    [math.nan, 3],
                    [13.0, 5],
                    [14.0, 6],
                ],
                columns=["a", "timestamp"],
            )
        )

        op = MovingStandardDeviationOperator(
            input=input_data.node(),
            window_length=1.1,
            sampling=node_lib.input_node([]),
        )
        instance = MovingStandardDeviationNumpyImplementation(op)

        sampling_data = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [1],
                    [2],
                    [2.5],
                    [3],
                    [3.5],
                    [4],
                    [5],
                    [6],
                ],
                columns=["timestamp"],
            )
        )

        output = instance(input=input_data, sampling=sampling_data)

        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [math.nan, 1],
                    [0, 2],
                    [0, 2.5],
                    [0, 3],
                    [math.nan, 3.5],
                    [math.nan, 4],
                    [0, 5],
                    [0.5, 6],
                ],
                columns=["a", "timestamp"],
            )
        )

        self.assertEqual(repr(output), repr({"output": expected_output}))


if __name__ == "__main__":
    absltest.main()
