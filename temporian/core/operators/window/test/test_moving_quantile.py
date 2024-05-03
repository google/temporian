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
from absl.testing.parameterized import parameters
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import f32, f64, assertOperatorResult, i32, i64
from temporian.core.data.duration import shortest


class QuantileTest(TestCase):
    @parameters(
        {  # empty f32
            "timestamps": [],
            "feature": f32([]),
            "window": 5.0,
            "output_feature": f32([]),
            "quantile": 0.5,
        },
        {  # empty f64
            "timestamps": [],
            "feature": f64([]),
            "window": 5.0,
            "output_feature": f64([]),
            "quantile": 0.5,
        },
        {  # normal
            "timestamps": [1, 2, 3, 5, 20],
            "feature": f32([0, 1, 2, 3, 4]),
            "window": 5.0,
            "output_feature": f32([0.0, 0.5, 1.0, 1.5, 4.0]),
            "quantile": 0.5,
        },
        {  # normal float 64
            "timestamps": [1, 2, 3, 5, 20],
            "feature": f64([0, 1, 2, 3, 4]),
            "window": 5.0,
            "output_feature": f64([0.0, 0.5, 1.0, 1.5, 4.0]),
            "quantile": 0.5,
        },
        {  # normal int32
            "timestamps": [1, 2, 3, 5, 20],
            "feature": i32([0, 1, 2, 3, 4]),
            "window": 5.0,
            "output_feature": f32([0.0, 0.5, 1.0, 1.5, 4.0]),
            "quantile": 0.5,
        },
        {  # normal int32
            "timestamps": [1, 2, 3, 5, 20],
            "feature": i64([0, 1, 2, 3, 4]),
            "window": 5.0,
            "output_feature": f32([0.0, 0.5, 1.0, 1.5, 4.0]),
            "quantile": 0.5,
        },
        {  # w nan
            "timestamps": [1, 1.5, 2, 5, 20],
            "feature": f32([10, nan, nan, 13, 14]),
            "window": 1.0,
            "output_feature": f32([10.0, 10.0, nan, 13.0, 14.0]),
            "quantile": 0.5,
        },
        {  # other quantiles
            "timestamps": [1, 2, 3, 5, 20],
            "feature": f32([0, 1, 2, 3, 4]),
            "window": 5.0,
            "output_feature": f32([0.0, 0.0, 0.5, 1.0, 4.0]),
            "quantile": 1 / 3,
        },
        {  # other quantiles
            "timestamps": [1, 2, 3, 5, 20],
            "feature": f32([0, 1, 2, 3, 4]),
            "window": 5.0,
            "output_feature": f32([0.0, 1.0, 2.0, 2.5, 4.0]),
            "quantile": 0.75,
        },
    )
    def test_without_sampling(
        self,
        timestamps,
        feature,
        window,
        output_feature,
        quantile,
    ):
        evset = event_set(timestamps=timestamps, features={"a": feature})

        expected = event_set(
            timestamps=timestamps,
            features={"a": output_feature},
            same_sampling_as=evset,
        )

        result = evset.moving_quantile(window_length=window, quantile=quantile)
        assertOperatorResult(self, result, expected)

    def test_numpy_equivalence(self):
        size = 1_000
        max_jump = 10
        w = 10
        q = 0.8
        np.random.seed(0)

        timestamps = np.random.random_integers(
            low=1, high=max_jump, size=size
        ).cumsum()
        feature = np.random.normal(size=size).cumsum()

        evset = event_set(timestamps=timestamps, features={"feature": feature})

        np_quantiles = []
        for i in timestamps:
            window = feature[(timestamps <= i) & (timestamps > i - w)]
            np_quantiles.append(
                np.quantile(window, q, method="averaged_inverted_cdf")
            )

        expected = event_set(
            timestamps=timestamps,
            features={"feature": np.array(np_quantiles)},
            same_sampling_as=evset,
        )

        result = evset.moving_quantile(window_length=w, quantile=q)

        assertOperatorResult(self, result, expected)

    @parameters(
        {  # empty f32
            "timestamps": [],
            "feature": f32([]),
            "window": 5.0,
            "sampling_timestamps": [],
            "output_feature": f32([]),
            "quantile": 0.5,
        },
        {  # empty f64
            "timestamps": [],
            "feature": f64([]),
            "window": 5.0,
            "sampling_timestamps": [],
            "output_feature": f64([]),
            "quantile": 0.5,
        },
        {  # normal
            "timestamps": [1, 2, 3, 5, 6],
            "feature": f32([10, 11, 12, 13, 14]),
            "window": 3.0,
            "sampling_timestamps": [-1.0, 1.0, 1.1, 3.0, 3.5, 6.0, 10.0],
            "output_feature": f32([nan, 10.0, 10.0, 11.0, 11.0, 13.5, nan]),
            "quantile": 0.5,
        },
        {  # w nan
            "timestamps": [1, 2, 3, 5, 6],
            "feature": f32([nan, 11, nan, 13, 14]),
            "window": 1.0,
            "sampling_timestamps": [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            "output_feature": f32([nan, 11.0, 11.0, nan, nan, nan, 13.0, 14]),
            "quantile": 0.5,
        },
    )
    def test_with_sampling(
        self,
        timestamps,
        feature,
        window,
        sampling_timestamps,
        output_feature,
        quantile,
    ):
        evset = event_set(timestamps=timestamps, features={"a": feature})

        sampling = event_set(timestamps=sampling_timestamps)

        expected = event_set(
            timestamps=sampling_timestamps,
            features={"a": output_feature},
            same_sampling_as=sampling,
        )

        result = evset.moving_quantile(
            window_length=window, sampling=sampling, quantile=quantile
        )
        assertOperatorResult(self, result, expected)

    @parameters(
        {  # normal
            "timestamps": [0, 1, 2, 3, 5, 20],
            "feature": f32([nan, 10, 11, 12, 13, 14]),
            "variable_window": f64([1, 1, 1.5, 0.5, 3.5, 20]),
            "output_feature": f32([nan, 10, 10.5, 12, 12, 12]),
            "quantile": 0.5,
        },
        {  # invalid values
            "timestamps": [0, 1, 2, 3, 5, 6, 20],
            "feature": [nan, 10, 11, 12, 13, 14, 15],
            "variable_window": [1, -20, 3, 0, 10, nan, 19],
            "output_feature": [nan, nan, 10.5, nan, 11.5, nan, 13],
            "quantile": 0.5,
        },
        {  # repeated ts
            "timestamps": [0, 2, 2, 2, 2, 5],
            "feature": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "variable_window": [1, 3, 0.5, np.inf, -1, 5],
            "output_feature": [10, 12, 12.5, 12, nan, 13],
            "quantile": 0.5,
        },
        {  # repeated ts same winlen
            "timestamps": [2, 2, 2, 2],
            "feature": f64([10, 11, 12, 13]),
            "variable_window": f64([0, 1, 1, 2]),
            "output_feature": [nan, 11.5, 11.5, 11.5],
            "quantile": 0.5,
        },
        {  # empty
            "timestamps": f64([]),
            "feature": f64([]),
            "variable_window": f64([]),
            "output_feature": f64([]),
            "quantile": 0.5,
        },
    )
    def test_with_variable_winlen_same_sampling(
        self,
        timestamps,
        feature,
        variable_window,
        output_feature,
        quantile,
    ):
        evset = event_set(timestamps=timestamps, features={"a": feature})

        window = event_set(
            timestamps=timestamps,
            features={"a": variable_window},
            same_sampling_as=evset,
        )

        expected = event_set(
            timestamps=timestamps,
            features={"a": output_feature},
            same_sampling_as=evset,
        )

        result = evset.moving_quantile(window_length=window, quantile=quantile)
        assertOperatorResult(self, result, expected)

    @parameters(
        {  # normal
            "timestamps": [0, 1, 2, 3, 5, 20],
            "feature": [nan, 10, 11, 12, 13, 14],
            "window_timestamps": [-1, 1, 4, 19, 20, 20],
            "variable_window": [10, 0.5, 2.5, 19, 16, np.inf],
            "output_feature": [nan, 10, 11.5, 11.5, 13.5, 12],
            "quantile": 0.5,
        },
        {  # repeated ts
            "timestamps": [0, 1, 2, 3, 5, 20],
            "feature": [nan, 10, 11, 12, 13, 14],
            "window_timestamps": [20, 20, 20, 20, 20, 20],
            "variable_window": [16, 0.001, np.inf, 0, 1, 19],
            "output_feature": [13.5, 14, 12, nan, 14, 12.5],
            "quantile": 0.5,
        },
        {  # shortest duration
            "timestamps": [1.999999, 2],
            "feature": [10.0, 11.0],
            "window_timestamps": [2, 2, 2, 2],
            "variable_window": [1, 0.001, shortest, 0],
            "output_feature": [10.5, 10.5, 11, nan],
            "quantile": 0.5,
        },
        {  # invalid values
            "timestamps": [0, 1, 2, 3, 5, 20],
            "feature": [nan, 10, 11, 12, 13, 14],
            "window_timestamps": [2, 2, 5, 5, 20, 20],
            "variable_window": [1, -10, 3, 0, nan, 19],
            "output_feature": [11, nan, 12.5, nan, nan, 12.5],
            "quantile": 0.5,
        },
        {  # repeated ts same winlen
            "timestamps": [0, 1, 2, 3],
            "feature": f64([10, 11, 12, 13]),
            "window_timestamps": [2, 2, 2, 2],
            "variable_window": f64([0, 1, 1, 2]),
            "output_feature": f64([nan, 12, 12, 11.5]),
            "quantile": 0.5,
        },
        {  # empty arrays
            "timestamps": f64([1]),
            "feature": f64([10]),
            "variable_window": f64([]),
            "window_timestamps": f64([]),
            "output_feature": f64([]),
            "quantile": 0.5,
        },
    )
    def test_with_variable_winlen_diff_sampling(
        self,
        timestamps,
        feature,
        variable_window,
        window_timestamps,
        output_feature,
        quantile,
    ):
        evset = event_set(timestamps=timestamps, features={"a": feature})

        window = event_set(
            timestamps=window_timestamps,
            features={"a": variable_window},
        )

        expected = event_set(
            timestamps=window_timestamps,
            features={"a": output_feature},
            same_sampling_as=window,
        )

        result = evset.moving_quantile(window_length=window, quantile=quantile)
        assertOperatorResult(self, result, expected)

    def test_with_index(self):
        evset = event_set(
            timestamps=[1, 2, 3, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            features={
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
                "a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            },
            indexes=["x", "y"],
        )
        expected = event_set(
            timestamps=[1, 2, 3, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            features={
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
                "a": [10.0, 10.5, 11.0, 13.0, 13.5, 14.0, 16.0, 16.5, 17.0],
            },
            indexes=["x", "y"],
            same_sampling_as=evset,
        )
        result = evset.moving_quantile(window_length=5.0, quantile=0.5)
        assertOperatorResult(self, result, expected)

    def test_error_input_bytes(self):
        evset = event_set([1, 2], {"f": ["A", "B"]})
        with self.assertRaisesRegex(
            ValueError,
            "moving_quantile requires the input EventSet to contain",
        ):
            _ = evset.moving_quantile(1, quantile=0.5)


if __name__ == "__main__":
    absltest.main()
