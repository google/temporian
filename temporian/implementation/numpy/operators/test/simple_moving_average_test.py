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

from absl.testing import absltest
import math

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from temporian.core.operators.window.simple_moving_average import (
    SimpleMovingAverageOperator,
)
from temporian.implementation.numpy.operators.window.simple_moving_average import (
    SimpleMovingAverageNumpyImplementation,
    operators_cc,
)
from temporian.core.data import node as node_lib
from temporian.implementation.numpy.data.io import pd_dataframe_to_event_set


def _f64(l):
    return np.array(l, np.float64)


def _f32(l):
    return np.array(l, np.float32)


nan = math.nan
cc_sma = operators_cc.simple_moving_average


class SimpleMovingAverageOperatorTest(absltest.TestCase):
    def test_cc_empty(self):
        a_f64 = _f64([])
        a_f32 = _f32([])
        assert_array_equal(cc_sma(a_f64, a_f32, 5.0), a_f32)
        assert_array_equal(cc_sma(a_f64, a_f64, 5.0), a_f64)
        assert_array_equal(
            cc_sma(a_f64, a_f32, a_f64, 5.0),
            a_f32,
        )
        assert_array_equal(
            cc_sma(a_f64, a_f64, a_f64, 5.0),
            a_f64,
        )

    def test_cc_wo_sampling(self):
        assert_array_equal(
            cc_sma(
                _f64([1, 2, 3, 5, 20]),
                _f32([10, 11, 12, 13, 14]),
                5.0,
            ),
            _f32([10.0, 10.5, 11.0, 11.5, 14.0]),
        )

    def test_cc_w_sampling(self):
        assert_array_equal(
            cc_sma(
                _f64([1, 2, 3, 5, 6]),
                _f32([10, 11, 12, 13, 14]),
                _f64([-1.0, 1.0, 1.1, 3.0, 3.5, 6.0, 10.0]),
                3.0,
            ),
            _f32([nan, 10.0, 10.0, 11.0, 11.0, 13.5, nan]),
        )

    def test_cc_w_nan_wo_sampling(self):
        assert_array_equal(
            cc_sma(
                _f64([1, 1.5, 2, 5, 20]),
                _f32([10, nan, nan, 13, 14]),
                1.0,
            ),
            _f32([10.0, 10.0, nan, 13.0, 14.0]),
        )

    def test_cc_w_nan_w_sampling(self):
        assert_array_equal(
            cc_sma(
                _f64([1, 2, 3, 5, 6]),
                _f32([nan, 11, nan, 13, 14]),
                _f64([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]),
                1.0,
            ),
            _f32([nan, 11.0, 11.0, nan, nan, nan, 13.0, 14]),
        )

    def test_flat(self):
        """A simple event set."""

        evset = pd_dataframe_to_event_set(
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

        op = SimpleMovingAverageOperator(
            input=evset.node(),
            window_length=5.0,
            sampling=None,
        )
        op.outputs["output"].check_same_sampling(evset.node())

        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = SimpleMovingAverageNumpyImplementation(op)
        output = instance.call(input=evset)

        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [10.0, 20.0, 1],
                    [10.5, 20.5, 2],
                    [11.0, 21.0, 3],
                    [11.5, 21.5, 5],
                    [14.0, 24.0, 20],
                ],
                columns=["a", "b", "timestamp"],
            )
        )

        self.assertEqual(repr(output), repr({"output": expected_output}))

    def test_with_index(self):
        """Indexed Event sets."""

        evset = pd_dataframe_to_event_set(
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

        op = SimpleMovingAverageOperator(
            input=evset.node(),
            window_length=5.0,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = SimpleMovingAverageNumpyImplementation(op)

        output = instance.call(input=evset)
        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    ["X1", "Y1", 10.0, 1],
                    ["X1", "Y1", 10.5, 2],
                    ["X1", "Y1", 11.0, 3],
                    ["X2", "Y1", 13.0, 1.1],
                    ["X2", "Y1", 13.5, 2.1],
                    ["X2", "Y1", 14.0, 3.1],
                    ["X2", "Y2", 16.0, 1.2],
                    ["X2", "Y2", 16.5, 2.2],
                    ["X2", "Y2", 17.0, 3.2],
                ],
                columns=["x", "y", "a", "timestamp"],
            ),
            index_names=["x", "y"],
        )

        self.assertEqual(output["output"], expected_output)

    def test_with_sampling(self):
        """Event sets with user provided sampling."""

        evset = pd_dataframe_to_event_set(
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

        sampling_node = node_lib.source_node([])
        op = SimpleMovingAverageOperator(
            input=evset.node(),
            window_length=3.0,
            sampling=sampling_node,
        )
        op.outputs["output"].check_same_sampling(sampling_node)

        self.assertEqual(
            op.list_matching_io_samplings(), [("sampling", "output")]
        )
        instance = SimpleMovingAverageNumpyImplementation(op)

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

        output = instance.call(input=evset, sampling=sampling_data)
        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [nan, -1.0],
                    [10.0, 1.0],
                    [10.0, 1.1],
                    [11.0, 3.0],
                    [11.0, 3.5],
                    [13.5, 6.0],
                    [nan, 10.0],
                ],
                columns=["a", "timestamp"],
            )
        )

        self.assertEqual(output["output"], expected_output)

    def test_with_nan(self):
        """The input features contains nan values."""

        evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [nan, 1],
                    [11.0, 2],
                    [nan, 3],
                    [13.0, 5],
                    [14.0, 6],
                ],
                columns=["a", "timestamp"],
            )
        )

        op = SimpleMovingAverageOperator(
            input=evset.node(),
            window_length=1.0,
            sampling=node_lib.source_node([]),
        )
        instance = SimpleMovingAverageNumpyImplementation(op)

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

        output = instance.call(input=evset, sampling=sampling_data)
        expected_output = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [nan, 1],
                    [11.0, 2],
                    [11.0, 2.5],
                    [nan, 3],
                    [nan, 3.5],
                    [nan, 4],
                    [13.0, 5],
                    [14.0, 6],
                ],
                columns=["a", "timestamp"],
            )
        )

        self.assertEqual(output["output"], expected_output)


if __name__ == "__main__":
    absltest.main()
