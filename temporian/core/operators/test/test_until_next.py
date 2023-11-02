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

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class UntilNextTest(absltest.TestCase):
    def test_with_timeout(self):
        a = event_set(timestamps=[0, 10, 11, 20, 30])
        b = event_set(timestamps=[1, 12, 21, 22, 42])

        result = a.until_next(sampling=b, timeout=5)

        expected = event_set(
            timestamps=[1, 12, 12, 21, 35],
            features={
                "until_next": [1, 2, 1, 1, math.nan],
            },
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_no_sampling(self):
        a = event_set(timestamps=[0], features={"x": ["a"]}, indexes=["x"])
        b = event_set(timestamps=[0], features={"x": ["b"]}, indexes=["x"])
        result = a.until_next(sampling=b, timeout=5)

        expected = event_set(
            timestamps=[5],
            features={"x": ["a"], "until_next": [math.nan]},
            indexes=["x"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_timeout_negative(self):
        evset = event_set([])
        sampling = event_set([])

        with self.assertRaisesRegex(
            ValueError, "A duration should be a strictly"
        ):
            evset.until_next(sampling=sampling, timeout=-5)

    def test_timeout_non_finite(self):
        evset = event_set([])
        sampling = event_set([])

        with self.assertRaisesRegex(
            ValueError, "Timeout should be finite. Instead, got "
        ):
            evset.until_next(sampling=sampling, timeout=math.inf)


if __name__ == "__main__":
    absltest.main()
