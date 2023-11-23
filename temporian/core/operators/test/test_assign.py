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

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class AssignTest(parameterized.TestCase):
    def test_basic(self):
        timestamps = [1, 1, 2, 3, 4]

        evset_1 = event_set(
            timestamps=timestamps,
            features={
                "f": [1, 2, 3, 4, 5],
            },
        )
        evset_2 = event_set(
            timestamps=timestamps,
            features={
                "f": [21, 22, 23, 24, 25],
            },
            same_sampling_as=evset_1,
        )

        result = evset_1.assign(f2=evset_2)

        expected = event_set(
            timestamps=timestamps,
            features={
                "f": [1, 2, 3, 4, 5],
                "f2": [21, 22, 23, 24, 25],
            },
            same_sampling_as=evset_1,
        )

        assertOperatorResult(self, result, expected)

    def test_multi(self):
        timestamps = [1, 2, 3, 4, 5]

        evset_1 = event_set(
            timestamps=timestamps,
            features={
                "f": [1, 2, 3, 4, 5],
            },
        )
        evset_2 = event_set(
            timestamps=timestamps,
            features={
                "f": [21, 22, 23, 24, 25],
            },
            same_sampling_as=evset_1,
        )

        evset_3 = event_set(
            timestamps=timestamps,
            features={
                "f": [31, 32, 33, 34, 35],
            },
            same_sampling_as=evset_1,
        )
        result = evset_1.assign(f2=evset_2, f3=evset_3)

        expected = event_set(
            timestamps=timestamps,
            features={
                "f": [1, 2, 3, 4, 5],
                "f2": [21, 22, 23, 24, 25],
                "f3": [31, 32, 33, 34, 35],
            },
            same_sampling_as=evset_1,
        )
        assertOperatorResult(self, result, expected)

    def test_multi_features(self):
        timestamps = [1, 2, 3, 4, 5]

        evset_1 = event_set(
            timestamps=timestamps,
            features={
                "f": [1, 2, 3, 4, 5],
            },
        )
        evset_2 = event_set(
            timestamps=timestamps,
            features={
                "f": [21, 22, 23, 24, 25],
                "g": [31, 32, 33, 34, 35],
            },
            same_sampling_as=evset_1,
        )

        with self.assertRaisesRegex(
            ValueError, "The assigned EventSets must have a single feature"
        ):
            result = evset_1.assign(f2=evset_2)

        result = evset_1.assign(g2=evset_2["g"])

        expected = event_set(
            timestamps=timestamps,
            features={
                "f": [1, 2, 3, 4, 5],
                "g2": [31, 32, 33, 34, 35],
            },
            same_sampling_as=evset_1,
        )
        assertOperatorResult(self, result, expected)

    def test_overwrite(self):
        timestamps = [1, 2, 3, 4, 5]

        evset_1 = event_set(
            timestamps=timestamps,
            features={
                "f": [1, 2, 3, 4, 5],
            },
        )
        evset_2 = event_set(
            timestamps=timestamps,
            features={
                "f": [21, 22, 23, 24, 25],
            },
            same_sampling_as=evset_1,
        )

        result = evset_1.assign(f=evset_2)

        expected = event_set(
            timestamps=timestamps,
            features={
                "f": [21, 22, 23, 24, 25],
            },
            same_sampling_as=evset_1,
        )
        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
