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


class WhereTest(TestCase):
    def setUp(self):
        self.evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "cond": [False, True, False, True, False, False],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
        )

    def test_both_single_values(self):
        on_true = "hi"
        on_false = "goodbye"

        result = self.evset.where(on_true=on_true, on_false=on_false)

        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "cond": [
                    on_false,
                    on_true,
                    on_false,
                    on_true,
                    on_false,
                    on_false,
                ],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )

        assertOperatorResult(self, result, expected)

    def test_both_evsets(self):
        sources = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "on_true": [5, 6, 7, 8, 9, 10],
                "on_false": [-5, -6, -7, -8, -9, -10],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )
        on_true = sources["on_true"]
        on_false = sources["on_false"]

        result = self.evset.where(on_true=on_true, on_false=on_false)

        # SetUp() condition: [False, True, False, True, False, False]
        expected_output = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "cond": [-5, 6, -7, 8, -9, -10],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, result, expected_output)

    def test_true_evset_false_single_value(self):
        on_false = -10
        on_true = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "on_true": [5, 6, 7, 8, 9, 10],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )

        result = self.evset.where(on_true=on_true, on_false=on_false)

        # SetUp() condition: [False, True, False, True, False, False]
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "cond": [-10, 6, -10, 8, -10, -10],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )

        assertOperatorResult(self, result, expected)

    def test_true_single_val_false_evset(self):
        on_true = 10
        on_false = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "on_true": [-5, -6, -7, -8, -9, -10],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )

        result = self.evset.where(on_true=on_true, on_false=on_false)

        # SetUp() condition: [False, True, False, True, False, False]
        expected_output = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "cond": [-5, 10, -7, 10, -9, -10],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )

        assertOperatorResult(self, result, expected_output)

    def test_dtype_mismatch_single_values(self):
        with self.assertRaisesRegex(ValueError, "should have the same dtype"):
            self.evset.where(on_true="A string", on_false=5)

    def test_dtype_mismatch_evsets(self):
        sources = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "on_true": [5, 6, 7, 8, 9, 10],
                "on_false": ["a", "b", "c", "d", "e", "f"],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )
        on_true = sources["on_true"]  # int64
        on_false = sources["on_false"]  # string

        with self.assertRaisesRegex(ValueError, "should have the same dtype"):
            self.evset.where(on_true=on_true, on_false=on_false)

    def test_dtype_mismatch_evset_to_single_value(self):
        source_evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "on_true": [5, 6, 7, 8, 9, 10],
                "idx": ["A", "A", "A", "B", "B", "B"],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )
        source_str = "A string"

        with self.assertRaisesRegex(ValueError, "should have the same dtype"):
            self.evset.where(on_true=source_evset, on_false=source_str)

        # Reverse on_true/on_false order
        with self.assertRaisesRegex(ValueError, "should have the same dtype"):
            self.evset.where(on_true=source_str, on_false=source_evset)

    def test_non_boolean_input(self):
        # Non-boolean condition
        non_boolean = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "cond": [5, 6, 7, 8, 9, 10],
            },
        )
        with self.assertRaisesRegex(
            ValueError, "Input should have only 1 boolean feature"
        ):
            non_boolean.where(on_true=1, on_false=0)

    def test_multiple_feats(self):
        multi_feats = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "cond": [True] * 6,
                "f1": [0] * 6,
                "f2": [1] * 6,
            },
        )

        # Check input
        with self.assertRaisesRegex(
            ValueError, "Input should have only 1 boolean feature"
        ):
            multi_feats.where(on_true=1, on_false=0)

        # Check on_true
        with self.assertRaisesRegex(ValueError, "should have only 1 feature"):
            multi_feats["cond"].where(on_true=multi_feats, on_false=0)

        # Check on_false
        with self.assertRaisesRegex(ValueError, "should have only 1 feature"):
            multi_feats["cond"].where(on_true=1, on_false=multi_feats)


if __name__ == "__main__":
    absltest.main()
