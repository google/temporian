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

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class LogicalTest(absltest.TestCase):
    """Test logical ops: | (OR) & (AND) ^ (XOR)"""

    def setUp(self):
        self.evset_1 = event_set(
            timestamps=[1, 2, 3, 4], features={"x": [True, False, True, False]}
        )
        self.evset_2 = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [True, False, False, True]},
            same_sampling_as=self.evset_1,
        )

    def test_correct_and(self) -> None:
        """Test correct AND operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [True, False, False, False]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(self, self.evset_1 & self.evset_2, expected)

    def test_correct_or(self) -> None:
        """Test correct OR operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [True, False, True, True]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(self, self.evset_1 | self.evset_2, expected)

    def test_correct_xor(self) -> None:
        """Test correct XOR operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [False, False, True, True]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(self, self.evset_1 ^ self.evset_2, expected)


if __name__ == "__main__":
    absltest.main()
