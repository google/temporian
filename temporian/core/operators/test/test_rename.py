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


class RenameTest(TestCase):
    def setUp(self):
        self.evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11]},
        )

    def test_rename_single_feature_with_str(self):
        result = self.evset.rename("b")

        expected = event_set(
            timestamps=[1, 2],
            features={"b": [10, 11]},
            same_sampling_as=self.evset,
        )

        assertOperatorResult(self, result, expected)

    def test_rename_single_feature_with_dict(self):
        result = self.evset.rename({"a": "b"})

        expected = event_set(
            timestamps=[1, 2],
            features={"b": [10, 11]},
            same_sampling_as=self.evset,
        )

        assertOperatorResult(self, result, expected)

    def test_rename_multiple_features(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2], "c": [100, 101]},
        )
        result = evset.rename({"a": "d", "b": "e"})

        expected = event_set(
            timestamps=[1, 2],
            features={"d": [10, 11], "e": [1, 2], "c": [100, 101]},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_rename_single_index_with_str(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
            indexes=["b"],
        )
        result = evset.rename(indexes="c")

        expected = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "c": [1, 2]},
            indexes=["c"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_rename_single_index_with_dict(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
            indexes=["b"],
        )
        result = evset.rename(indexes={"b": "c"})

        expected = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "c": [1, 2]},
            indexes=["c"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_rename_multiple_indexes(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
            indexes=["a", "b"],
        )
        result = evset.rename(indexes={"a": "c", "b": "d"})

        expected = event_set(
            timestamps=[1, 2],
            features={"c": [10, 11], "d": [1, 2]},
            indexes=["c", "d"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_rename_feature_with_empty_str(self):
        with self.assertRaisesRegex(
            ValueError, "Cannot rename to an empty string"
        ):
            self.evset.rename("")

    def test_rename_feature_with_empty_str_with_dict(self):
        with self.assertRaisesRegex(
            ValueError, "Cannot rename to an empty string"
        ):
            self.evset.rename({"a": ""})

    def test_rename_feature_with_non_str_object(self):
        with self.assertRaises(ValueError):
            self.evset.rename({"a": 1})

    def test_rename_feature_with_non_existent_feature(self) -> None:
        with self.assertRaises(KeyError):
            self.evset.rename({"sales_1": "costs"})

    def test_rename_feature_with_duplicated_new_feature_names(self) -> None:
        with self.assertRaises(ValueError):
            self.evset.rename({"sales": "new_sales", "costs": "new_sales"})

    def test_rename_index_with_empty_str(self) -> None:
        """Test renaming index with empty string."""
        with self.assertRaises(ValueError):
            self.evset.rename(indexes="")

    def test_rename_index_with_empty_str_with_dict(self) -> None:
        with self.assertRaises(ValueError):
            self.evset.rename(indexes={"sales": ""})

    def test_rename_index_with_non_existent_index(self) -> None:
        with self.assertRaises(KeyError):
            self.evset.rename(indexes={"xxx": "yyy"})

    def test_rename_index_with_duplicated_new_indexes(self) -> None:
        with self.assertRaises(ValueError):
            self.evset.rename(
                indexes={"store_id": "new_sales", "sales": "new_sales"}
            )

    def test_rename_feature_and_index_with_same_name(self) -> None:
        """Test renaming feature and index with same name."""
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
            indexes=["b"],
        )

        with self.assertRaises(ValueError):
            evset.rename(
                indexes={"b": "a"},
            )

    def test_rename_feature_and_index_inverting_name(self) -> None:
        """Test renaming feature and index with same name."""
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
            indexes=["b"],
        )

        result = evset.rename(
            features={"a": "b"},
            indexes={"b": "a"},
        )

        expected = event_set(
            timestamps=[1, 2],
            features={"b": [10, 11], "a": [1, 2]},
            indexes=["a"],
        )
        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_single_string_when_multiple_features(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
        )

        with self.assertRaisesRegex(
            ValueError,
            (
                "Cannot apply rename operator with a single rename string when "
                "the EventSet contains multiple features. Pass a dictionary "
                "of rename strings instead."
            ),
        ):
            evset.rename("c")


if __name__ == "__main__":
    absltest.main()
