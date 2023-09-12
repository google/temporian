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
from unittest.mock import patch

import numpy as np
import pandas as pd
from absl.testing import absltest
from numpy.testing import assert_array_equal

from temporian.core.operators.window.moving_sum import (
    MovingSumOperator,
)
from temporian.implementation.numpy.operators.window.moving_sum import (
    MovingSumNumpyImplementation,
    operators_cc,
)
from temporian.core.data import node as node_lib
from temporian.io.pandas import from_pandas


def _f64(l):
    return np.array(l, np.float64)


def _f32(l):
    return np.array(l, np.float32)


nan = math.nan


class MovingSumOperatorTest(absltest.TestCase):
    def test_cc_wo_sampling(self):
        assert_array_equal(
            operators_cc.moving_sum(
                evset_timestamps=_f64([1, 2, 3, 5, 20]),
                evset_values=_f32([10, nan, 12, 13, 14]),
                window_length=5.0,
            ),
            _f32([10.0, 10.0, 22.0, 35.0, 14.0]),
        )

    def test_cc_wo_sampling_w_variable_winlen(self):
        assert_array_equal(
            operators_cc.moving_sum(
                evset_timestamps=_f64([0, 1, 2, 3, 5, 20]),
                evset_values=_f32([nan, 10, 11, 12, 13, 14]),
                window_length=_f64([1, 1, 1.5, 0.5, 3.5, 20]),
            ),
            _f32([0, 10, 21, 12, 36, 60]),
        )

    def test_cc_w_sampling_w_variable_winlen(self):
        assert_array_equal(
            operators_cc.moving_sum(
                evset_timestamps=_f64([0, 1, 2, 3, 5, 20]),
                evset_values=_f32([nan, 10, 11, 12, 13, 14]),
                sampling_timestamps=_f64([-1, 1, 4, 19, 20, 20]),
                window_length=_f64([10, 0.5, 2.5, 19, 16, np.inf]),
            ),
            _f32([0, 10, 23, 46, 27, 60]),
        )

    def test_flat(self):
        """A simple event set."""

        evset = from_pandas(
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

        op = MovingSumOperator(
            input=evset.node(),
            window_length=5.0,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = MovingSumNumpyImplementation(op)

        output = instance(input=evset)

        expected_output = from_pandas(
            pd.DataFrame(
                [
                    [10.0, 20.0, 1],
                    [21.0, 41.0, 2],
                    [33.0, 63.0, 3],
                    [46.0, 86.0, 5],
                    [14.0, 24.0, 20],
                ],
                columns=["a", "b", "timestamp"],
            )
        )

        self.assertEqual(repr(output), repr({"output": expected_output}))

    def test_with_index(self):
        """Indexed event set."""

        evset = from_pandas(
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
            indexes=["x", "y"],
        )

        op = MovingSumOperator(
            input=evset.node(),
            window_length=5.0,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = MovingSumNumpyImplementation(op)

        output = instance(input=evset)

        expected_output = from_pandas(
            pd.DataFrame(
                [
                    ["X1", "Y1", 10.0, 1],
                    ["X1", "Y1", 21.0, 2],
                    ["X1", "Y1", 33.0, 3],
                    ["X2", "Y1", 13.0, 1.1],
                    ["X2", "Y1", 27.0, 2.1],
                    ["X2", "Y1", 42.0, 3.1],
                    ["X2", "Y2", 16.0, 1.2],
                    ["X2", "Y2", 33.0, 2.2],
                    ["X2", "Y2", 51.0, 3.2],
                ],
                columns=["x", "y", "a", "timestamp"],
            ),
            indexes=["x", "y"],
        )

        self.assertEqual(output["output"], expected_output)

    def test_with_sampling(self):
        """Event sets with user provided sampling."""

        evset = from_pandas(
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

        op = MovingSumOperator(
            input=evset.node(),
            window_length=3.1,
            sampling=node_lib.input_node([]),
        )
        self.assertEqual(
            op.list_matching_io_samplings(), [("sampling", "output")]
        )
        instance = MovingSumNumpyImplementation(op)

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
                    [10.0, 1.0],
                    [10.0, 1.1],
                    [33.0, 3.0],
                    [33.0, 3.5],
                    [39.0, 6.0],
                    [0, 10.0],
                ],
                columns=["a", "timestamp"],
            )
        )

        self.assertEqual(output["output"], expected_output)

    def test_with_variable_winlen_different_sampling(self):
        evset = from_pandas(
            pd.DataFrame(
                [
                    [1, 10.0],
                    [2, 11.0],
                    [3, 12.0],
                    [5, 13.0],
                    [6, 14.0],
                ],
                columns=["timestamp", "a"],
            )
        )
        window_length = from_pandas(
            pd.DataFrame(
                [
                    [2, 0.5],
                    [5.5, 3],
                    [10, 8.5],
                ],
                columns=["timestamp", "length"],
            ),
        )

        op = MovingSumOperator(
            input=evset.node(),
            window_length=window_length.node(),
        )
        instance = MovingSumNumpyImplementation(op)

        output = instance(input=evset, window_length=window_length)

        expected_output = from_pandas(
            pd.DataFrame(
                [
                    [2, 11.0],
                    [5.5, 25.0],
                    [10, 50],
                ],
                columns=["timestamp", "a"],
            )
        )

        self.assertEqual(output["output"], expected_output)

    @patch(
        "temporian.implementation.numpy.operators.window.moving_sum.operators_cc.moving_sum"
    )
    def test_with_variable_winlen_same_sampling_uses_correct_cpp_impl(
        self, cpp_moving_sum_mock
    ):
        """Checks that the no-sampling version of cpp code is called when
        passing a variable window_length with same sampling as the input."""
        evset = from_pandas(
            pd.DataFrame([[1, 10.0]], columns=["timestamp", "a"])
        )
        window_length = from_pandas(
            pd.DataFrame([[1, 1.0]], columns=["timestamp", "length"]),
            same_sampling_as=evset,
        )

        op = MovingSumOperator(
            input=evset.node(),
            window_length=window_length.node(),
        )
        instance = MovingSumNumpyImplementation(op)

        instance(input=evset, window_length=window_length)

        # sampling_timestamps not passed
        cpp_moving_sum_mock.assert_called_once_with(
            evset_timestamps=evset.data[()].timestamps,
            evset_values=evset.data[()].features[0],
            window_length=window_length.data[()].features[0],
        )

    def test_with_sampling_and_variable_winlen_error(self):
        evset = from_pandas(
            pd.DataFrame([[1, 10.0]], columns=["timestamp", "a"])
        )
        window_length = from_pandas(
            pd.DataFrame([[2, 0.5]], columns=["timestamp", "length"]),
        )

        with self.assertRaisesRegex(
            ValueError,
            (
                "`sampling` cannot be specified if a variable `window_length`"
                " is specified"
            ),
        ):
            MovingSumOperator(
                input=evset.node(),
                window_length=window_length.node(),
                sampling=window_length[[]].node(),
            )

    def test_wo_sampling_w_variable_winlen(self):
        evset = from_pandas(
            pd.DataFrame(
                [
                    [1, 10.0],
                    [2, 11.0],
                    [3, 12.0],
                    [5, 13.0],
                    [6, 14.0],
                ],
                columns=["timestamp", "a"],
            )
        )
        window_length = from_pandas(
            pd.DataFrame(
                [
                    [2, 0.5],
                    [5.5, 3],
                    [10, 8.5],
                ],
                columns=["timestamp", "length"],
            ),
        )

        op = MovingSumOperator(
            input=evset.node(),
            window_length=window_length.node(),
        )
        instance = MovingSumNumpyImplementation(op)

        output = instance(input=evset, window_length=window_length)

        expected_output = from_pandas(
            pd.DataFrame(
                [
                    [2, 11.0],
                    [5.5, 25.0],
                    [10, 50],
                ],
                columns=["timestamp", "a"],
            )
        )

        self.assertEqual(output["output"], expected_output)

    def test_with_nan(self):
        """The input features contains nan values."""

        evset = from_pandas(
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

        op = MovingSumOperator(
            input=evset.node(),
            window_length=1.1,
            sampling=node_lib.input_node([]),
        )
        instance = MovingSumNumpyImplementation(op)

        sampling_data = from_pandas(
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

        output = instance(input=evset, sampling=sampling_data)

        expected_output = from_pandas(
            pd.DataFrame(
                [
                    [0, 1],
                    [11.0, 2],
                    [11.0, 2.5],
                    [11.0, 3],
                    [0, 3.5],
                    [0, 4],
                    [13.0, 5],
                    [27.0, 6],
                ],
                columns=["a", "timestamp"],
            )
        )

        self.assertEqual(output["output"], expected_output)

    def test_cumsum(self):
        """Infinite window length (aka: cumsum function)"""

        input_data = from_pandas(
            pd.DataFrame(
                [
                    ["X1", "Y1", 10.0, 1.0, 1],
                    ["X1", "Y1", 11.0, -1, 2],
                    ["X1", "Y1", 12.0, 2, 3],
                    ["X2", "Y1", 13.0, -3, 1.1],
                    ["X2", "Y1", 14.0, -8, 2.1],
                    ["X2", "Y1", 15.0, 0, 3.1],
                    ["X2", "Y2", 16.0, 5, 1.2],
                    ["X2", "Y2", 17.0, 3, 2.2],
                    ["X2", "Y2", 18.0, -1, 3.2],
                ],
                columns=["x", "y", "a", "b", "timestamp"],
            ),
            indexes=["x", "y"],
        )

        op = MovingSumOperator(
            input=input_data.node(),
            window_length=np.inf,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = MovingSumNumpyImplementation(op)

        output = instance(input=input_data)

        expected_output = from_pandas(
            pd.DataFrame(
                [
                    ["X1", "Y1", 10.0, 1.0, 1],
                    ["X1", "Y1", 21.0, 0, 2],
                    ["X1", "Y1", 33.0, 2, 3],
                    ["X2", "Y1", 13.0, -3, 1.1],
                    ["X2", "Y1", 27.0, -11, 2.1],
                    ["X2", "Y1", 42.0, -11, 3.1],
                    ["X2", "Y2", 16.0, 5, 1.2],
                    ["X2", "Y2", 33.0, 8, 2.2],
                    ["X2", "Y2", 51.0, 7, 3.2],
                ],
                columns=["x", "y", "a", "b", "timestamp"],
            ),
            indexes=["x", "y"],
        )

        self.assertEqual(output["output"], expected_output)


if __name__ == "__main__":
    absltest.main()
