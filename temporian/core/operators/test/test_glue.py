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
from absl.testing.parameterized import TestCase
from temporian.core.operators.glue import glue

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, f32


class GlueTest(TestCase):
    def test_basic(self):
        timestamps = [1, 1, 2, 3, 4]

        evset_1 = event_set(
            timestamps=timestamps,
            features={
                "x": ["a", "a", "a", "a", "b"],
                "f1": [10, 11, 12, 13, 14],
            },
            indexes=["x"],
        )
        evset_2 = event_set(
            timestamps=timestamps,
            features={
                "x": ["a", "a", "a", "a", "b"],
                "f2": [20, 21, 22, 23, 24],
                "f3": [30, 31, 32, 33, 34],
            },
            indexes=["x"],
            same_sampling_as=evset_1,
        )
        evset_3 = event_set(
            timestamps=timestamps,
            features={
                "x": ["a", "a", "a", "a", "b"],
                "f4": [40, 41, 42, 43, 44],
            },
            indexes=["x"],
            same_sampling_as=evset_1,
        )

        result = glue(evset_1, evset_2, evset_3)

        expected = event_set(
            timestamps=timestamps,
            features={
                "x": ["a", "a", "a", "a", "b"],
                "f1": [10, 11, 12, 13, 14],
                "f2": [20, 21, 22, 23, 24],
                "f3": [30, 31, 32, 33, 34],
                "f4": [40, 41, 42, 43, 44],
            },
            indexes=["x"],
            same_sampling_as=evset_1,
        )

        assertOperatorResult(self, result, expected)

    def test_non_matching_sampling(self):
        with self.assertRaisesRegex(
            ValueError,
            "Arguments should have the same sampling.",
        ):
            evset_1 = event_set([])
            evset_2 = event_set([0])
            glue(evset_1, evset_2)

    def test_duplicate_feature(self):
        with self.assertRaisesRegex(
            ValueError,
            'Feature "a" is defined in multiple input EventSetNodes',
        ):
            evset_1 = event_set([], features={"a": []})
            evset_2 = event_set(
                [], features={"a": []}, same_sampling_as=evset_1
            )
            glue(evset_1, evset_2)

    def test_order_unchanged(self):
        """Tests that input evsets' order is kept.

        Regression test for failing case where glue misordered its inputs when
        more than 10, because of sorted() being called over a list where
        "input_10"  was interpreted as less than "input_2".
        """
        evset_0 = event_set(
            timestamps=[1],
            features={"f0": [1]},
        )
        evset_1 = evset_0.rename("f1")
        evset_2 = evset_0.rename("f2")
        evset_3 = evset_0.rename("f3")
        evset_4 = evset_0.rename("f4")
        evset_5 = evset_0.rename("f5")
        evset_6 = evset_0.rename("f6")
        evset_7 = evset_0.rename("f7")
        evset_8 = evset_0.rename("f8")

        # Test that alphabetical order is not used
        evset_9 = evset_0.rename("a")

        evset_10 = event_set(
            timestamps=[1],
            features={"f10": f32([1])},
            same_sampling_as=evset_0,
        )

        result = glue(
            evset_0,
            evset_1,
            evset_2,
            evset_3,
            evset_4,
            evset_5,
            evset_6,
            evset_7,
            evset_8,
            evset_9,
            evset_10,
        )

        expected = event_set(
            [1],
            features={
                "f0": [1],
                "f1": [1],
                "f2": [1],
                "f3": [1],
                "f4": [1],
                "f5": [1],
                "f6": [1],
                "f7": [1],
                "f8": [1],
                "a": [1],
                "f10": f32([1]),
            },
            same_sampling_as=evset_0,
        )

        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
