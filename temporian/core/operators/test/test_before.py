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

from datetime import datetime

import numpy as np
from absl.testing import absltest
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class BeforeTest(TestCase):
    def test_basic(self):
        evset = event_set(timestamps=[1, 2, 3], features={"x": [4, 5, 6]})

        result = evset.before(2)

        expected = event_set(
            timestamps=[1],
            features={"x": [4]},
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_empty(self):
        evset = event_set(timestamps=[1, 2, 3], features={"x": [4, 5, 6]})

        result = evset.before(0)

        # use numpy arrays to maintain the correct dtype on empty arrays
        expected = event_set(
            timestamps=np.array([], dtype=np.int64),
            features={
                "x": np.array([], dtype=np.int64),
            },
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_all(self):
        evset = event_set(timestamps=[1, 2, 3], features={"x": [4, 5, 6]})

        result = evset.before(4)

        expected = event_set(timestamps=[1, 2, 3], features={"x": [4, 5, 6]})

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_floats_and_int(self):
        evset = event_set(
            timestamps=[1.1, 1.9, 2.1],
            features={"x": [4, 5, 6]},
        )

        result = evset.before(2)

        expected = event_set(
            timestamps=[1.1, 1.9],
            features={"x": [4, 5]},
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

        evset = event_set(
            timestamps=[1, 2, 3],
            features={"x": [4, 5, 6]},
        )

        result = evset.before(2.9999)

        expected = event_set(
            timestamps=[1, 2],
            features={"x": [4, 5]},
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_datetime(self):
        evset = event_set(
            timestamps=[
                datetime(2023, 11, 16, 10, 15, 00),
                datetime(2023, 11, 16, 10, 16, 00),
                datetime(2023, 11, 16, 10, 17, 00),
            ],
            features={"x": [4, 5, 6]},
        )

        result = evset.before(datetime(2023, 11, 16, 10, 16, 00))

        expected = event_set(
            timestamps=[
                datetime(2023, 11, 16, 10, 15, 00),
            ],
            features={"x": [4]},
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

        evset = event_set(
            timestamps=[1.1, 1.9, 2.1],
            features={"x": [4, 5, 6]},
        )

        with self.assertRaisesRegex(ValueError, "unix"):
            result = evset.before(datetime(2023, 11, 16, 10, 16, 00))


if __name__ == "__main__":
    absltest.main()
