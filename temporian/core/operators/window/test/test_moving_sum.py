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

from math import nan

import numpy as np
from absl.testing import absltest
from absl.testing.parameterized import TestCase, parameters

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import f32, f64, assertOperatorResult


class MovingSumTest(TestCase):
    def test_without_sampling(self):
        timestamps = f64([1, 2, 3, 5, 20])
        evset = event_set(
            timestamps=timestamps, features={"a": f32([10, nan, 12, 13, 14])}
        )

        expected = event_set(
            timestamps=timestamps,
            features={"a": f32([10.0, 10.0, 22.0, 35.0, 14.0])},
            same_sampling_as=evset,
        )

        result = evset.moving_sum(window_length=5.0)
        assertOperatorResult(self, result, expected)

    def test_without_sampling_many_features(self):
        timestamps = [1, 2, 3, 5, 20]
        evset = event_set(
            timestamps=timestamps,
            features={
                "a": [10.0, 11.0, 12.0, 13.0, 14.0],
                "b": [20.0, 21.0, 22.0, 23.0, 24.0],
            },
        )

        expected = event_set(
            timestamps=timestamps,
            features={
                "a": [10.0, 21.0, 33.0, 46.0, 14.0],
                "b": [20.0, 41.0, 63.0, 86.0, 24.0],
            },
            same_sampling_as=evset,
        )

        result = evset.moving_sum(window_length=5.0)
        assertOperatorResult(self, result, expected)

    def test_without_sampling_with_index(self):
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

        expected = event_set(
            timestamps=timestamps,
            features={
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
                "a": [10.0, 21.0, 33.0, 13.0, 27.0, 42.0, 16.0, 33.0, 51.0],
            },
            indexes=["x", "y"],
            same_sampling_as=evset,
        )

        result = evset.moving_sum(window_length=5.0)
        assertOperatorResult(self, result, expected)

    @parameters(
        {  # normal
            "timestamps": f64([1, 2, 3, 5, 6]),
            "feature": [10.0, 11.0, 12.0, 13.0, 14.0],
            "window_length": 3.1,
            "sampling_timestamps": [-1.0, 1.0, 1.1, 3.0, 3.5, 6.0, 10.0],
            "output_feature": [0.0, 10.0, 10.0, 33.0, 33.0, 39.0, 0.0],
        },
        {  # w nan
            "timestamps": f64([1, 2, 3, 5, 6]),
            "feature": [nan, 11.0, nan, 13.0, 14.0],
            "window_length": 1.1,
            "sampling_timestamps": [1, 2, 2.5, 3, 3.5, 4, 5, 6],
            "output_feature": [0.0, 11.0, 11.0, 11.0, 0.0, 0.0, 13.0, 27.0],
        },
    )
    def test_with_sampling(
        self,
        timestamps,
        feature,
        window_length,
        sampling_timestamps,
        output_feature,
    ):
        evset = event_set(
            timestamps=timestamps,
            features={"a": feature},
        )
        sampling = event_set(timestamps=sampling_timestamps)

        expected = event_set(
            timestamps=sampling_timestamps,
            features={"a": output_feature},
            same_sampling_as=sampling,
        )

        result = evset.moving_sum(
            window_length=window_length, sampling=sampling
        )
        assertOperatorResult(self, result, expected)

    def test_with_variable_winlen_same_sampling(self):
        timestamps = f64([0, 1, 2, 3, 5, 20])
        evset = event_set(
            timestamps=timestamps,
            features={"a": f32([nan, 10, 11, 12, 13, 14])},
        )

        window = event_set(
            timestamps=timestamps,
            features={"a": f64([1, 1, 1.5, 0.5, 3.5, 20])},
            same_sampling_as=evset,
        )

        expected = event_set(
            timestamps=timestamps,
            features={"a": f32([0, 10, 21, 12, 36, 60])},
            same_sampling_as=evset,
        )

        result = evset.moving_sum(window_length=window)
        assertOperatorResult(self, result, expected)

    def test_with_variable_winlen_diff_sampling(self):
        window_timestamps = f64([-1, 1, 4, 19, 20, 20])
        window_length = f64([10, 0.5, 2.5, 19, 16, np.inf])

        evset = event_set(
            timestamps=f64([0, 1, 2, 3, 5, 20]),
            features={"a": f32([nan, 10, 11, 12, 13, 14])},
        )

        window = event_set(
            timestamps=window_timestamps,
            features={"a": window_length},
        )

        expected = event_set(
            timestamps=window_timestamps,
            features={"a": f32([0, 10, 23, 46, 27, 60])},
            same_sampling_as=window,
        )

        result = evset.moving_sum(window_length=window)
        assertOperatorResult(self, result, expected)

    def test_error_input_bytes(self):
        evset = event_set([1, 2], {"f": ["A", "B"]})
        with self.assertRaisesRegex(
            ValueError,
            "moving_sum requires the input EventSet to contain numerical",
        ):
            _ = evset.moving_sum(1)


if __name__ == "__main__":
    absltest.main()
