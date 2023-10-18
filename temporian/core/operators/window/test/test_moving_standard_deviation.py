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
from math import nan

import numpy as np
from absl.testing import absltest
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, f32, f64


class MovingStandardDeviationAverageTest(TestCase):
    def test_basic(self):
        timestamps = [1, 2, 3, 5, 20]
        evset = event_set(
            timestamps=timestamps, features={"a": f32([10, nan, 12, 13, 14])}
        )

        result = evset.moving_standard_deviation(window_length=5.0)

        expected = event_set(
            timestamps=timestamps,
            features={"a": f32([0.0, 0.0, 1.0, 1.247219, 0.0])},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_with_index(self):
        timestamps = [1, 2, 3, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2]
        evset = event_set(
            timestamps=timestamps,
            features={
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
                "a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            },
            indexes=["x", "y"],
        )

        result = evset.moving_standard_deviation(window_length=5.0)

        expected = event_set(
            timestamps=timestamps,
            features={
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
                "a": [
                    0,
                    0.5,
                    math.sqrt(2 / 3),
                    0,
                    0.5,
                    math.sqrt(2 / 3),
                    0,
                    0.5,
                    math.sqrt(2 / 3),
                ],
            },
            indexes=["x", "y"],
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_w_sampling(self):
        timestamps = [1, 2, 3, 5, 6]
        evset = event_set(
            timestamps=timestamps, features={"a": f64([10, 11, 12, 13, 14])}
        )
        sampling_timestamps = [-1.0, 1.0, 1.1, 3.0, 3.5, 6.0, 10.0]
        sampling = event_set(timestamps=sampling_timestamps)

        result = evset.moving_standard_deviation(
            window_length=3.1, sampling=sampling
        )

        expected = event_set(
            timestamps=sampling_timestamps,
            features={
                "a": [
                    math.nan,
                    0,
                    0,
                    math.sqrt(2 / 3),
                    math.sqrt(2 / 3),
                    math.sqrt(2 / 3),
                    math.nan,
                ]
            },
            same_sampling_as=sampling,
        )

        assertOperatorResult(self, result, expected)

    def test_with_nan(self):
        timestamps = [1, 2, 3, 5, 6]
        evset = event_set(
            timestamps=timestamps,
            features={"a": [math.nan, 11.0, math.nan, 13.0, 14.0]},
        )
        sampling_timestamps = [1, 2, 2.5, 3, 3.5, 4, 5, 6]
        sampling = event_set(timestamps=sampling_timestamps)

        result = evset.moving_standard_deviation(
            window_length=1.1, sampling=sampling
        )

        expected = event_set(
            timestamps=sampling_timestamps,
            features={"a": [math.nan, 0, 0, 0, math.nan, math.nan, 0, 0.5]},
            same_sampling_as=sampling,
        )

        assertOperatorResult(self, result, expected)

    def test_wo_sampling(self):
        timestamps = [1, 2, 3, 5, 20]
        evset = event_set(
            timestamps=timestamps, features={"a": f32([10, nan, 12, 13, 14])}
        )

        result = evset.moving_standard_deviation(window_length=5.0)

        expected = event_set(
            timestamps=timestamps,
            features={"a": f32([0.0, 0.0, 1.0, 1.247219, 0.0])},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_wo_sampling_w_variable_winlen(self):
        timestamps = [0, 1, 2, 3, 5, 20]
        evset = event_set(
            timestamps=timestamps,
            features={"a": f32([nan, 10, 11, 12, 13, 14])},
        )

        winlen = event_set(
            timestamps=timestamps,
            features={"a": f64([1, 1, 1.5, 0.5, 3.5, 20])},
            same_sampling_as=evset,
        )

        result = evset.moving_standard_deviation(window_length=winlen)

        expected = event_set(
            timestamps=timestamps,
            features={"a": f32([nan, 0, 0.5, 0, 0.8164965, 1.4142135])},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_w_sampling_w_variable_winlen(self):
        timestamps = [0, 1, 2, 3, 5, 20]
        evset = event_set(
            timestamps=timestamps,
            features={"a": f32([nan, 10, 11, 12, 13, 14])},
        )

        sampling_timestamps = [-1, 1, 4, 19, 20, 20]

        winlen = event_set(
            timestamps=sampling_timestamps,
            features={"a": f64([10, 0.5, 2.5, 19, 16, np.inf])},
        )

        result = evset.moving_standard_deviation(window_length=winlen)

        expected = event_set(
            timestamps=sampling_timestamps,
            features={"a": f32([nan, 0, 0.5, 1.1180339, 0.5, 1.4142135])},
            same_sampling_as=winlen,
        )

        assertOperatorResult(self, result, expected)

    def test_error_input_int(self):
        evset = event_set([1, 2], {"f": [1, 2]})
        with self.assertRaisesRegex(
            ValueError,
            "moving_standard_deviation requires the input EventSet to contain",
        ):
            _ = evset.moving_standard_deviation(1)

    def test_error_input_bytes(self):
        evset = event_set([1, 2], {"f": ["A", "B"]})
        with self.assertRaisesRegex(
            ValueError,
            "moving_standard_deviation requires the input EventSet to contain",
        ):
            _ = evset.moving_standard_deviation(1)


if __name__ == "__main__":
    absltest.main()
