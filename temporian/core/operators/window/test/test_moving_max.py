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
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, f32


class MovingMaxTest(TestCase):
    def test_basic(self):
        timestamps = [0, 1, 2, 3, 5, 20]
        evset = event_set(
            timestamps=timestamps,
            features={"a": f32([nan, 10, nan, 12, 13, 14])},
        )

        result = evset.moving_max(window_length=3.5)

        expected = event_set(
            timestamps=timestamps,
            features={"a": f32([nan, 10, 10, 12, 13, 14])},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_w_sampling(self):
        timestamps = [0, 1, 2, 3, 5, 20]
        evset = event_set(
            timestamps=timestamps,
            features={"a": f32([nan, 10, nan, 12, 13, 14])},
        )
        sampling_timestamps = [-1, 3, 40]
        sampling = event_set(timestamps=sampling_timestamps)

        result = evset.moving_max(window_length=3.5, sampling=sampling)

        expected = event_set(
            timestamps=sampling_timestamps,
            features={"a": f32([nan, 12, nan])},
            same_sampling_as=sampling,
        )

        assertOperatorResult(self, result, expected)

    def test_wo_sampling_w_variable_winlen(self):
        timestamps = [0, 1, 2, 3, 5, 20]
        evset = event_set(
            timestamps=timestamps,
            features={"a": [nan, 0, 10, 5, 1, 2]},
        )
        winlen = event_set(
            timestamps=timestamps,
            features={"a": [1, 1, 1.5, 0.5, 3.5, 0]},
            same_sampling_as=evset,
        )

        result = evset.moving_max(window_length=winlen)

        expected = event_set(
            timestamps=timestamps,
            features={"a": [nan, 0, 10, 5, 10, nan]},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_w_sampling_w_variable_winlen(self):
        timestamps = [0, 1, 2, 3, 5, 20]
        evset = event_set(
            timestamps=timestamps,
            features={"a": [nan, 0, 10, 5, 1, 2]},
        )
        sampling_timestamps = [-1, 1, 4, 19, 20, 20]
        winlen = event_set(
            timestamps=sampling_timestamps,
            features={"a": [10, 10, 2.5, 19, 0.001, np.inf]},
        )

        result = evset.moving_max(window_length=winlen)

        expected = event_set(
            timestamps=sampling_timestamps,
            features={"a": [nan, 0, 10, 10, 2, 10]},
            same_sampling_as=winlen,
        )

        assertOperatorResult(self, result, expected)

    def test_error_input_bytes(self):
        evset = event_set([1, 2], {"f": ["A", "B"]})
        with self.assertRaisesRegex(
            ValueError,
            "moving_max requires the input EventSet to contain numerical",
        ):
            _ = evset.moving_max(1)


if __name__ == "__main__":
    absltest.main()
