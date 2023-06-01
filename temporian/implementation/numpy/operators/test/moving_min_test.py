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
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from temporian.core.operators.window.moving_min import MovingMinOperator
from temporian.implementation.numpy.operators.window.moving_min import (
    MovingMinNumpyImplementation,
    operators_cc,
)
from temporian.implementation.numpy.data.event_set import EventSet


def _f64(l):
    return np.array(l, np.float64)


def _f32(l):
    return np.array(l, np.float32)


def _i32(l):
    return np.array(l, np.int32)


nan = math.nan


class MovingMinOperatorTest(absltest.TestCase):
    def test_cc_wo_sampling(self):
        assert_array_equal(
            operators_cc.moving_min(
                _f64([0, 1, 2, 3, 5, 6, 6, 20]),
                _f32([nan, 10, nan, 12, 13, 14, 2, 15]),
                3.5,
            ),
            _f32([nan, 10, 10, 10, 12, 2, 2, 15]),
        )

    def test_cc_w_sampling(self):
        assert_array_equal(
            operators_cc.moving_min(
                _f64([0, 1, 2, 3, 5, 20]),
                _f32([nan, 10, nan, 12, 13, 14]),
                _f64([-1, 3, 40]),
                3.5,
            ),
            _f32([nan, 10, nan]),
        )

    def test_flat(self):
        """A simple time sequence."""

        input_data = EventSet.from_dataframe(
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

        op = MovingMinOperator(
            input=input_data.node(),
            window_length=5.0,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("input", "output")])
        instance = MovingMinNumpyImplementation(op)

        output = instance.call(input=input_data)

        expected_output = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [10.0, 20.0, 1],
                    [0.0, 20.0, 2],
                    [0.0, 0.0, 3],
                    [0.0, 0.0, 5],
                    [14.0, 24.0, 20],
                ],
                columns=["a", "b", "timestamp"],
            )
        )

        self.assertEqual(repr(output), repr({"output": expected_output}))


if __name__ == "__main__":
    absltest.main()
