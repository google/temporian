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


class DropIndexTest(TestCase):
    def setUp(self) -> None:
        self.timestamps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.features = {
            "a": ["A", "A", "A", "B", "B", "B", "B", "B"],
            "b": [0, 0, 0, 0, 0, 1, 1, 1],
            "c": [1, 1, 1, 2, 2, 2, 2, 3],
            "d": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        }
        self.evset = event_set(
            timestamps=self.timestamps,
            features=self.features,
            indexes=["b", "c"],
        )

    def test_drop_all(self) -> None:
        expected = event_set(
            timestamps=self.timestamps,
            # Old indexes are now the last features
            features={
                "a": self.features["a"],
                "d": self.features["d"],
                "b": self.features["b"],
                "c": self.features["c"],
            },
        )

        result = self.evset.drop_index()

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_drop_single_first(self) -> None:
        expected = event_set(
            timestamps=self.timestamps,
            # Old indexes are now the last features
            features={
                "c": self.features["c"],
                "a": self.features["a"],
                "d": self.features["d"],
                "b": self.features["b"],
            },
            indexes=["c"],
        )

        result = self.evset.drop_index("b")

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_drop_single_second(self) -> None:
        expected = event_set(
            timestamps=self.timestamps,
            # Old indexes are now the last features
            features={
                "b": self.features["b"],
                "a": self.features["a"],
                "d": self.features["d"],
                "c": self.features["c"],
            },
            indexes=["b"],
        )

        result = self.evset.drop_index("c")

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_drop_single_keep_false(self) -> None:
        expected = event_set(
            timestamps=self.timestamps,
            # Old indexes are now the last features
            features={
                "b": self.features["b"],
                "a": self.features["a"],
                "d": self.features["d"],
            },
            indexes=["b"],
        )

        result = self.evset.drop_index("c", keep=False)

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_str_index(self) -> None:
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "a": ["A", "A", "A", "B", "B", "C"],
                "b": [1, 1, 2, 2, 3, 3],
            },
            indexes=["a", "b"],
        )
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "a": ["A", "A", "A", "B", "B", "C"],
                "b": [1, 1, 2, 2, 3, 3],
            },
            indexes=["a"],
        )

        result = evset.drop_index("b")

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_wrong_index(self):
        with self.assertRaisesRegex(ValueError, "x is not an index in"):
            self.evset.drop_index("x")

    def test_empty_list(self):
        with self.assertRaisesRegex(
            ValueError, "Cannot specify empty list as `indexes` argument"
        ):
            self.evset.drop_index([])


if __name__ == "__main__":
    absltest.main()
