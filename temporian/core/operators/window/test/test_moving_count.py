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

import numpy as np
from absl.testing import absltest
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, i32


class MovingCountTest(TestCase):
    def test_basic(self):
        timestamps = [1, 2, 3, 5, 20]
        evset = event_set(timestamps=timestamps)

        result = evset.moving_count(window_length=5.0)

        expected = event_set(
            timestamps=timestamps,
            features={"count": i32([1, 2, 3, 4, 1])},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_w_sampling(self):
        timestamps = [0, 1, 2, 3, 5, 20]
        evset = event_set(timestamps=timestamps)
        sampling_timestamps = [-1, 3, 40]
        sampling = event_set(timestamps=sampling_timestamps)

        result = evset.moving_count(window_length=3.5, sampling=sampling)

        expected = event_set(
            timestamps=sampling_timestamps,
            features={"count": i32([0, 4, 0])},
            same_sampling_as=sampling,
        )

        assertOperatorResult(self, result, expected)

    def test_wo_sampling_w_variable_winlen(self):
        timestamps = [1, 2, 3, 5, 20]
        evset = event_set(timestamps=timestamps)
        winlen = event_set(
            timestamps=timestamps,
            features={"a": [0, np.inf, 1.001, 5, 0.00001]},
            same_sampling_as=evset,
        )

        result = evset.moving_count(window_length=winlen)

        expected = event_set(
            timestamps=timestamps,
            features={"count": i32([0, 2, 2, 4, 1])},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_w_sampling_w_variable_winlen(self):
        timestamps = [1, 2, 3, 5, 20]
        evset = event_set(timestamps=timestamps)
        sampling_timestamps = [0, 1.5, 3.5, 3.5, 3.5, 20]
        winlen = event_set(
            timestamps=sampling_timestamps,
            features={"a": [1, 1, 1, 3, 0.5, 19.5]},
        )

        result = evset.moving_count(window_length=winlen)

        expected = event_set(
            timestamps=sampling_timestamps,
            features={"count": i32([0, 1, 1, 3, 0, 5])},
            same_sampling_as=winlen,
        )

        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
