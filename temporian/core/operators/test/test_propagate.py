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


class PropagateTest(TestCase):
    def test_basic(self):
        evset = event_set(
            timestamps=[1, 2, 3],
            features={
                "a": [1, 2, 3],
                "x": [1, 1, 2],
            },
            indexes=["x"],
        )
        sampling = event_set(
            timestamps=[1, 1, 1, 1],
            features={"x": [1, 1, 2, 2], "y": [1, 2, 1, 2]},
            indexes=["x", "y"],
        )

        result = evset.propagate(sampling)

        expected = event_set(
            timestamps=[1, 2, 1, 2, 3, 3],
            features={
                "a": [1, 2, 1, 2, 3, 3],
                "x": [1, 1, 1, 1, 2, 2],
                "y": [1, 1, 2, 2, 1, 2],
            },
            indexes=["x", "y"],
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_remove_index(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [1, 2], "x": [1, 2]},
            indexes=["x"],
        )
        sampling = event_set(
            timestamps=[3, 4],
            features={"x": [1, 1], "y": [1, 2]},
            indexes=["x", "y"],
        )

        result = evset.propagate(sampling)

        expected = event_set(
            timestamps=[1, 1],
            features={"a": [1, 1], "x": [1, 1], "y": [1, 2]},
            indexes=["x", "y"],
        )
        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_add_empty_index(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [1, 2], "x": [1, 2]},
            indexes=["x"],
        )
        sampling = event_set(
            timestamps=[3, 4, 5, 6],
            features={"x": [1, 1, 3, 3], "y": [1, 2, 4, 5]},
            indexes=["x", "y"],
        )

        result = evset.propagate(sampling)

        expected = event_set(
            timestamps=[1, 1, 1, 1],
            features={"a": [1, 1, 2, 2], "x": [1, 1, 3, 3], "y": [1, 2, 4, 5]},
            indexes=["x", "y"],
        )
        expected = expected.filter(expected["a"] != 2)
        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_wrong_index(self):
        evset = event_set([], features={"x": []}, indexes=["x"])
        sampling = event_set([], features={"y": []}, indexes=["y"])

        with self.assertRaisesRegex(
            ValueError,
            (
                "The indexes of input should be contained in the indexes of"
                " sampling"
            ),
        ):
            evset.propagate(sampling)

    def test_wrong_index_type(self):
        evset = event_set([0], features={"x": [1]}, indexes=["x"])
        sampling = event_set(
            [0], features={"x": ["a"], "y": [1]}, indexes=["x", "y"]
        )

        with self.assertRaisesRegex(
            ValueError,
            "However, the dtype is different",
        ):
            evset.propagate(sampling)


if __name__ == "__main__":
    absltest.main()
