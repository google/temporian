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
            "state_id": ["A", "A", "A", "B", "B", "B", "B", "B"],
            "store_id": [0, 0, 0, 0, 0, 1, 1, 1],
            "item_id": [1, 1, 1, 2, 2, 2, 2, 3],
            "sales": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        }
        self.evset = event_set(
            timestamps=self.timestamps,
            features=self.features,
            indexes=["store_id", "item_id"],
        )

    def test_drop_all(self) -> None:
        expected = event_set(
            timestamps=self.timestamps,
            # Old indexes are now the last features
            features={
                "state_id": self.features["state_id"],
                "sales": self.features["sales"],
                "store_id": self.features["store_id"],
                "item_id": self.features["item_id"],
            },
        )

        result = self.evset.drop_index()

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_drop_single_first(self) -> None:
        expected = event_set(
            timestamps=self.timestamps,
            # Old indexes are now the last features
            features={
                "item_id": self.features["item_id"],
                "state_id": self.features["state_id"],
                "sales": self.features["sales"],
                "store_id": self.features["store_id"],
            },
            indexes=["item_id"],
        )

        result = self.evset.drop_index("store_id")

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_drop_single_second(self) -> None:
        expected = event_set(
            timestamps=self.timestamps,
            # Old indexes are now the last features
            features={
                "store_id": self.features["store_id"],
                "state_id": self.features["state_id"],
                "sales": self.features["sales"],
                "item_id": self.features["item_id"],
            },
            indexes=["store_id"],
        )

        result = self.evset.drop_index("item_id")

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_drop_single_keep_false(self) -> None:
        expected = event_set(
            timestamps=self.timestamps,
            # Old indexes are now the last features
            features={
                "store_id": self.features["store_id"],
                "state_id": self.features["state_id"],
                "sales": self.features["sales"],
            },
            indexes=["store_id"],
        )

        result = self.evset.drop_index("item_id", keep=False)

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


if __name__ == "__main__":
    absltest.main()
