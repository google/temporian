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

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class SelectIndexValuesTest(TestCase):
    def setUp(self):
        self.evset = event_set(
            timestamps=[1, 2, 3],
            features={
                "a": [1.0, 2.0, 3.0],
                "b": [5, 6, 7],
                "c": ["A", "A", "B"],
            },
            indexes=["c"],
        )
        self.node = self.evset.node()

    def test_basic(self):
        result = self.evset.select_index_values([("A",)])

        expected = event_set(
            timestamps=[1, 2],
            features={
                "a": [1.0, 2.0],
                "b": [5, 6],
                "c": ["A", "A"],
            },
            indexes=["c"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_single_index_key_value(self):
        result = self.evset.select_index_values("B")

        expected = event_set(
            timestamps=[3],
            features={
                "a": [3.0],
                "b": [7],
                "c": ["B"],
            },
            indexes=["c"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_number(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            features={
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            indexes=["a"],
        )
        result = evset.select_index_values(number=3)

        self.assertEqual(len(result.data.keys()), 3)

    def test_number_larger_than_total(self):
        evset = event_set(
            timestamps=[1, 2, 3],
            features={
                "a": [1, 2, 3],
            },
            indexes=["a"],
        )
        result = evset.select_index_values(number=4)

        self.assertEqual(len(result.data.keys()), 3)

    def test_fraction_0(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            features={
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            indexes=["a"],
        )
        result = evset.select_index_values(fraction=0.0)

        self.assertEqual(len(result.data.keys()), 0)

    def test_fraction_rounds_down(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            features={
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            indexes=["a"],
        )
        result = evset.select_index_values(fraction=0.79)

        self.assertEqual(len(result.data.keys()), 7)

    def test_fraction_1(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            features={
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            indexes=["a"],
        )
        result = evset.select_index_values(fraction=1.0)

        self.assertEqual(len(result.data.keys()), 10)

    def test_fraction(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            features={
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            indexes=["a"],
        )
        result = evset.select_index_values(fraction=0.7)

        self.assertEqual(len(result.data.keys()), 7)

    def test_many_indexes_many_keys_change_order(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "b": [5, 6, 7, 8, 9, 10],
                "c": ["A", "A", "B", "B", "C", "C"],
                "d": [1, 2, 1, 2, 1, 2],
            },
            indexes=["c", "d"],
        )

        result = evset.select_index_values([("C", 1), ("A", 1), ("B", 2)])

        expected = event_set(
            timestamps=[5, 1, 4],
            features={
                "a": [5.0, 1.0, 4.0],
                "b": [9, 5, 8],
                "c": ["C", "A", "B"],
                "d": [1, 1, 2],
            },
            indexes=["c", "d"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_wrong_index_key(self):
        with self.assertRaisesRegex(
            ValueError, r"Index key '\(b'D',\)' not found in input EventSet."
        ):
            self.evset.select_index_values("D")

    def test_more_than_one_param(self):
        with self.assertRaisesRegex(
            ValueError,
            "Exactly one of keys, number or fraction must be provided.",
        ):
            self.evset.select_index_values("D", number=1)

    def test_no_params(self):
        with self.assertRaisesRegex(
            ValueError,
            "Exactly one of keys, number or fraction must be provided.",
        ):
            self.evset.select_index_values()

    def test_invalid_fraction(self):
        with self.assertRaisesRegex(
            ValueError,
            "fraction must be between 0 and 1.",
        ):
            self.evset.select_index_values(fraction=-1.0)

        with self.assertRaisesRegex(
            ValueError,
            "fraction must be between 0 and 1.",
        ):
            self.evset.select_index_values(fraction=1.01)

    def test_invalid_number(self):
        with self.assertRaisesRegex(
            ValueError,
            "number must be greater than 0.",
        ):
            self.evset.select_index_values(number=0)

        with self.assertRaisesRegex(
            ValueError,
            "number must be greater than 0.",
        ):
            self.evset.select_index_values(number=-1)


if __name__ == "__main__":
    absltest.main()
