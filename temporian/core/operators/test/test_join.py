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

import math

from absl.testing import absltest
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, f64


class JoinTest(TestCase):
    def test_base(self):
        evset_left = event_set(
            timestamps=[1, 2, 3, 5, 5, 6, 6],
            features={"a": [11, 12, 13, 14, 15, 16, 17]},
        )
        evset_right = event_set(
            timestamps=[1, 2, 4, 5, 5],
            features={"b": [21.0, 22.0, 23.0, 24.0, 25.0]},
        )

        result = evset_left.join(evset_right)

        expected = event_set(
            timestamps=[1, 2, 3, 5, 5, 6, 6],
            features={
                "a": [11, 12, 13, 14, 15, 16, 17],
                "b": [21.0, 22.0, math.nan, 24.0, 24.0, math.nan, math.nan],
            },
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_base_on(self):
        evset_left = event_set(
            timestamps=[1, 2, 2, 3, 4, 5],
            features={
                "a": [11, 12, 13, 14, 15, 16],
                "c": [0, 1, 2, 3, 4, 5],
            },
        )
        evset_right = event_set(
            timestamps=[1, 2, 2, 3, 4],
            features={
                "c": [0, 2, 1, 3, 5],
                "b": [11.0, 12.0, 13.0, 14.0, 15.0],
            },
        )

        result = evset_left.join(evset_right, on="c")

        expected = event_set(
            timestamps=[1, 2, 2, 3, 4, 5],
            features={
                "a": [11, 12, 13, 14, 15, 16],
                "c": [0, 1, 2, 3, 4, 5],
                "b": [11.0, 13.0, 12.0, 14.0, math.nan, math.nan],
            },
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_left(self):
        evset_1 = event_set([0], features={"a": [0]})
        evset_2 = event_set([0], features={"b": [0]})
        evset_1.join(evset_2)

    def test_left_on(self):
        evset_1 = event_set([0], {"a": [0], "x": [0]})
        evset_2 = event_set([0], {"b": [0], "x": [0]})
        evset_1.join(evset_2, on="x")

    def test_duplicated_feature(self):
        evset_1 = event_set([0], features={"a": [0]})
        evset_2 = event_set([0], features={"a": [0]})
        with self.assertRaisesRegex(ValueError, "is defined in both inputs"):
            evset_1.join(evset_2)

    def test_wrong_index(self):
        evset_1 = event_set([0], features={"a": [0]})
        evset_2 = event_set([0], features={"b": [0], "x": ["x"]}, indexes=["x"])
        with self.assertRaisesRegex(
            ValueError, "Arguments don't have the same index"
        ):
            evset_1.join(evset_2)

    def test_wrong_join(self):
        evset_1 = event_set([0], features={"a": [0]})
        evset_2 = event_set([0], features={"b": [0]})
        with self.assertRaisesRegex(ValueError, "Non supported join type"):
            evset_1.join(evset_2, how="non existing join")

    def test_missing_on(self):
        evset_1 = event_set([0], features={"a": [0]})
        evset_2 = event_set([0], features={"b": [0]})
        with self.assertRaisesRegex(ValueError, "does not exist in left"):
            evset_1.join(evset_2, on="c")

    def test_wrong_on_type(self):
        evset_1 = event_set([0], features={"a": [0], "c": f64([0])})
        evset_2 = event_set([0], features={"b": [0]})
        with self.assertRaisesRegex(ValueError, "Got float64 instead for left"):
            evset_1.join(evset_2, on="c")

    def test_same_sampling(self):
        evset_1 = event_set([0], features={"a": [0]})
        evset_2 = event_set([0], features={"b": [0]}, same_sampling_as=evset_1)
        with self.assertRaisesRegex(
            ValueError,
            "Both inputs have the same sampling. Use ",
        ):
            evset_1.join(evset_2)


if __name__ == "__main__":
    absltest.main()
