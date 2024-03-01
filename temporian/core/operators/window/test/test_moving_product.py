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


class MovingProductTest(TestCase):
    def test_without_sampling(self):
        timestamps = f64([1, 2, 3, 5, 6])
        evset = event_set(
            timestamps=timestamps, features={"a": f32([2, nan, 3, 0, 5])}
        )

        expected = event_set(
            timestamps=timestamps,
            features={"a": f32([2.0, 2.0, 3.0, 0.0, 0.0])},
            same_sampling_as=evset,
        )

        result = evset.moving_product(window_length=2.0)
        assertOperatorResult(self, result, expected)

    def test_with_zeros_and_nans(self):
        timestamps = f64([1, 2, 3, 4])
        evset = event_set(
            timestamps=timestamps, features={"a": f32([2.0, 0.0, nan, 3.0])}
        )

        expected = event_set(
            timestamps=timestamps,
            features={"a": f32([2.0, 0.0, 0.0, 3.0])},
            same_sampling_as=evset,
        )

        result = evset.moving_product(window_length=2.0)
        assertOperatorResult(self, result, expected)

    def test_empty_event_set(self):
        timestamps = f64([])
        evset = event_set(timestamps=timestamps, features={"a": f32([])})

        expected = event_set(
            timestamps=timestamps,
            features={"a": f32([])},
            same_sampling_as=evset,
        )

        result = evset.moving_product(window_length=2.0)
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
                "a": [10.0, 110.0, 132.0, 13.0, 14.0],
                "b": [20.0, 420.0, 462.0, 23.0, 24.0],
            },
            same_sampling_as=evset,
        )

        result = evset.moving_product(window_length=2.0)
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
                "a": [
                    10.0,
                    110.0,
                    1320.0,
                    13.0,
                    182.0,
                    2730.0,
                    16.0,
                    272.0,
                    4896.0,
                ],
            },
            indexes=["x", "y"],
            same_sampling_as=evset,
        )

        result = evset.moving_product(window_length=5.0)
        assertOperatorResult(self, result, expected)

    @parameters(
        {  # normal
            "timestamps": f64([1, 2, 3, 5, 6]),
            "feature": [10.0, 11.0, 12.0, 13.0, 14.0],
            "window_length": 3.1,
            "sampling_timestamps": [-1.0, 1.0, 1.1, 3.0, 3.5, 6.0, 10.0],
            "output_feature": [
                1.000e00,
                1.000e01,
                1.000e01,
                1.320e03,
                1.320e03,
                2.184e03,
                1.000e00,
            ],
        },
        {  # w nan
            "timestamps": f64([1, 2, 3, 5, 6]),
            "feature": [nan, 11.0, nan, 13.0, 14.0],
            "window_length": 1.1,
            "sampling_timestamps": [1, 2, 2.5, 3, 3.5, 4, 5, 6],
            "output_feature": [1.0, 11.0, 11.0, 11.0, 1.0, 1.0, 13.0, 182.0],
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

        result = evset.moving_product(
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
            features={"a": f32([1, 10, 110, 12, 1, 1])},
            same_sampling_as=evset,
        )

        result = evset.moving_product(window_length=window)
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
            features={"a": f32([1, 10, 132, 1, 182, 1])},
            same_sampling_as=window,
        )

        result = evset.moving_product(window_length=window)
        assertOperatorResult(self, result, expected)

    def test_error_input_int(self):
        evset = event_set([1, 2], {"f": [1, 2]})
        with self.assertRaisesRegex(
            ValueError,
            "moving_product requires the input EventSet to contain",
        ):
            _ = evset.moving_product(1)

    def test_error_input_bytes(self):
        evset = event_set([1, 2], {"f": ["A", "B"]})
        with self.assertRaisesRegex(
            ValueError,
            "moving_product requires the input EventSet to contain",
        ):
            _ = evset.moving_product(1)


if __name__ == "__main__":
    absltest.main()
