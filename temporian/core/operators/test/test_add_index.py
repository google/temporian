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
from temporian.test.utils import assertOperatorResult, i32


class AddIndexTest(TestCase):
    def setUp(self) -> None:
        self.timestamps = [0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, 0.3]
        self.features = {
            "state_id": ["A", "A", "A", "B", "B", "B", "B", "B"],
            "store_id": [0, 0, 0, 0, 0, 1, 1, 1],
            "item_id": [1, 1, 1, 2, 2, 2, 2, 3],
            "sales": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        }
        self.evset = event_set(
            timestamps=self.timestamps,
            features=self.features,
            indexes=["state_id"],
        )

    def test_add_index_single(self):
        result = self.evset.add_index("store_id")

        expected = event_set(
            timestamps=self.timestamps,
            features=self.features,
            indexes=["state_id", "store_id"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_add_index_multiple(self):
        result = self.evset.add_index(["store_id", "item_id"])

        expected = event_set(
            timestamps=self.timestamps,
            features=self.features,
            indexes=["state_id", "store_id", "item_id"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_set_index_single(self):
        result = self.evset.set_index("store_id")

        expected = event_set(
            timestamps=self.timestamps,
            # Old index is now the last feature
            features={
                **{k: v for k, v in self.features.items() if k != "state_id"},
                "state_id": self.features["state_id"],
            },
            indexes=["store_id"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_set_index_multiple(self):
        result = self.evset.set_index(["store_id", "item_id"])

        expected = event_set(
            timestamps=self.timestamps,
            # Old index is now the last feature
            features={
                **{k: v for k, v in self.features.items() if k != "state_id"},
                "state_id": self.features["state_id"],
            },
            indexes=["store_id", "item_id"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_set_index_multiple_change_order(self):
        common = {"features": {"a": [], "b": [], "c": []}, "timestamps": []}

        evset_abc = event_set(**common, indexes=["a", "b", "c"])
        evset_acb = event_set(**common, indexes=["a", "c", "b"])
        evset_cba = event_set(**common, indexes=["c", "b", "a"])
        evset_cab = event_set(**common, indexes=["c", "a", "b"])

        def test(src_evset, new_index, expected_evset):
            result = src_evset.set_index(new_index)
            assertOperatorResult(
                self, result, expected_evset, check_sampling=False
            )

        test(evset_abc, ["a", "b", "c"], evset_abc)
        test(evset_abc, ["a", "c", "b"], evset_acb)
        test(evset_abc, ["c", "b", "a"], evset_cba)
        test(evset_abc, ["c", "a", "b"], evset_cab)
        test(evset_cba, ["a", "b", "c"], evset_abc)

    def test_empty(self):
        evset = event_set([], features={"a": []})

        result = evset.add_index("a")

        self.assertEqual(len(result.data), 0)

    def test_int64(self):
        evset = event_set(
            [1, 2, 3, 4, 5, 6],
            features={"a": [1, 1, 1, 2, 2, 3], "b": [5, 5, 7, 1, 1, 4]},
        )

        result = evset.add_index(["a", "b"])

        self.assertEqual(len(result.data), 4)
        self.assertEqual(result.data[(1, 5)].timestamps.tolist(), [1, 2])
        self.assertEqual(result.data[(1, 7)].timestamps.tolist(), [3])
        self.assertEqual(result.data[(2, 1)].timestamps.tolist(), [4, 5])
        self.assertEqual(result.data[(3, 4)].timestamps.tolist(), [6])

    def test_int32(self):
        evset = event_set(
            [1, 2, 3, 4, 5, 6],
            features={
                "a": i32([1, 1, 1, 2, 2, 3]),
                "b": i32([5, 5, 7, 1, 1, 4]),
            },
        )

        result = evset.add_index(["a", "b"])

        self.assertEqual(len(result.data), 4)
        self.assertEqual(result.data[(1, 5)].timestamps.tolist(), [1, 2])
        self.assertEqual(result.data[(1, 7)].timestamps.tolist(), [3])
        self.assertEqual(result.data[(2, 1)].timestamps.tolist(), [4, 5])
        self.assertEqual(result.data[(3, 4)].timestamps.tolist(), [6])

    def test_str(self):
        evset = event_set(
            [1, 2, 3, 4, 5, 6],
            features={
                "a": ["A", "A", "A", "B", "B", "C"],
                "b": ["X", "X", "Y", "X", "X", "Z"],
            },
        )

        result = evset.add_index(["a", "b"])

        self.assertEqual(len(result.data), 4)
        self.assertEqual(result.data[(b"A", b"X")].timestamps.tolist(), [1, 2])
        self.assertEqual(result.data[(b"A", b"Y")].timestamps.tolist(), [3])
        self.assertEqual(result.data[(b"B", b"X")].timestamps.tolist(), [4, 5])
        self.assertEqual(result.data[(b"C", b"Z")].timestamps.tolist(), [6])

    def test_mix(self):
        evset = event_set(
            [1, 2, 3, 4, 5, 6],
            features={
                "a": [1, 1, 1, 2, 2, 3],
                "b": ["X", "X", "Y", "X", "X", "Z"],
            },
        )

        result = evset.add_index(["a", "b"])

        self.assertEqual(len(result.data), 4)
        self.assertEqual(result.data[(1, b"X")].timestamps.tolist(), [1, 2])
        self.assertEqual(result.data[(1, b"Y")].timestamps.tolist(), [3])
        self.assertEqual(result.data[(2, b"X")].timestamps.tolist(), [4, 5])
        self.assertEqual(result.data[(3, b"Z")].timestamps.tolist(), [6])


if __name__ == "__main__":
    absltest.main()
