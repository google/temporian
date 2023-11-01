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


from absl.testing import absltest, parameterized

import numpy as np

from temporian.core.data import duration
from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class FilterMovingCountTest(parameterized.TestCase):
    @parameterized.parameters(
        {"input_timestamps": [], "expected_timestamps": [], "win_length": 1.5},
        {
            "input_timestamps": [1, 2, 3],
            "expected_timestamps": [1, 3],
            "win_length": 1.5,
        },
        {
            "input_timestamps": [1, 1, 2, 3],
            "expected_timestamps": [1, 3],
            "win_length": 1.5,
        },
        {
            "input_timestamps": [0, 0, duration.shortest, 2, 3],
            "expected_timestamps": [0, duration.shortest, 2, 3],
            "win_length": duration.shortest,
        },
        {
            "input_timestamps": [1, 2, 3],
            "expected_timestamps": [1, 2, 3],
            "win_length": 1,
        },
        {
            "input_timestamps": [1, 2, 3],
            "expected_timestamps": [1, 3],
            "win_length": np.nextafter(1, 2),
        },
    )
    def test_base(self, input_timestamps, expected_timestamps, win_length):
        evset = event_set(input_timestamps)
        expected_output = event_set(expected_timestamps)
        result = evset.filter_moving_count(window_length=win_length)

        # NOTE: check_sampling=False still checks timestamps
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )


if __name__ == "__main__":
    absltest.main()
