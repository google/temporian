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

from absl.testing import absltest
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class SinceLastTest(TestCase):
    def test_no_sampling(self):
        ts = [1, 5, 8, 9, 1, 1, 2]
        x = [1, 1, 1, 1, 2, 2, 2]
        evset = event_set(
            timestamps=ts,
            features={"x": x},
            indexes=["x"],
        )

        result = evset.since_last()

        expected = event_set(
            timestamps=ts,
            features={"x": x, "since_last": [nan, 4, 3, 1, nan, 0, 1]},
            indexes=["x"],
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_no_sampling_2steps(self):
        ts = [1, 5, 8, 9, 1, 1, 2, 2, 2]
        x = [1, 1, 1, 1, 2, 2, 2, 2, 2]
        evset = event_set(
            timestamps=ts,
            features={"x": x},
            indexes=["x"],
        )

        result = evset.since_last(steps=2)

        expected = event_set(
            timestamps=ts,
            features={
                "x": x,
                "since_last": [nan, nan, 7, 4, nan, nan, 1, 1, 0],
            },
            indexes=["x"],
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_with_sampling(self):
        sampling_ts = [-1, 1, 1.5, 2, 2.1, 4, 5]

        evset = event_set(timestamps=[1, 2, 2, 4])
        sampling = event_set(timestamps=sampling_ts)

        result = evset.since_last(sampling=sampling)

        expected = event_set(
            timestamps=sampling_ts,
            features={"since_last": [nan, 0, 0.5, 0, 0.1, 0, 1]},
            same_sampling_as=sampling,
        )

        assertOperatorResult(self, result, expected)

    def test_with_sampling_2steps(self):
        sampling_ts = [-1, 1, 1.5, 2, 2, 2.1, 4, 5]

        evset = event_set(timestamps=[1, 2, 2, 4])
        sampling = event_set(timestamps=sampling_ts)

        result = evset.since_last(sampling=sampling, steps=2)

        expected = event_set(
            timestamps=sampling_ts,
            features={"since_last": [nan, nan, nan, 0, 0, 0.1, 2, 3]},
            same_sampling_as=sampling,
        )

        assertOperatorResult(self, result, expected)

    def test_negative_steps(self):
        evset = event_set([])

        with self.assertRaisesRegex(
            ValueError, "Number of steps must be greater than 0. Got"
        ):
            evset.since_last(steps=-1)


if __name__ == "__main__":
    absltest.main()
