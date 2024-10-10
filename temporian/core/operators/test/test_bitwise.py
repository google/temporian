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
from temporian.core.operators.binary import (
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    left_shift,
    right_shift,
)


class BitwiseTest(absltest.TestCase):
    """Test bitwise ops: | (OR) & (AND) ^ (XOR) << (LEFT SHIFT) >> (RIGHT SHIFT)"""

    def setUp(self):
        self.evset_1 = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [85, 170, 255, 51]},
        )
        self.evset_2 = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [2, 3, 5, 7]},
            same_sampling_as=self.evset_1,
        )

    def test_correct_and(self) -> None:
        """Test correct AND operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"bitwise_and_x_x": [0, 2, 5, 3]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(
            self, bitwise_and(self.evset_1, self.evset_2), expected
        )

    def test_correct_or(self) -> None:
        """Test correct OR operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"bitwise_or_x_x": [87, 171, 255, 55]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(
            self, bitwise_or(self.evset_1, self.evset_2), expected
        )

    def test_correct_xor(self) -> None:
        """Test correct XOR operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"bitwise_xor_x_x": [87, 169, 250, 52]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(
            self, bitwise_xor(self.evset_1, self.evset_2), expected
        )

    def test_correct_left_shift(self) -> None:
        """Test correct LEFT SHIFT operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"left_shift_x_x": [340, 1360, 8160, 6528]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(
            self, left_shift(self.evset_1, self.evset_2), expected
        )

    def test_correct_right_shift(self) -> None:
        """Test correct RIGHT SHIFT operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"right_shift_x_x": [21, 21, 7, 0]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(
            self, right_shift(self.evset_1, self.evset_2), expected
        )


if __name__ == "__main__":
    absltest.main()
