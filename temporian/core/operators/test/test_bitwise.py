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
from temporian.core.operators.binary.bitwise import bitwise_and, bitwise_or, bitwise_xor
from absl.testing import absltest
from absl.testing.parameterized import TestCase

from temporian.test.utils import assertOperatorResult, i32


class BitwiseTest(absltest.TestCase):
    """Test bitwise ops: | (OR) & (AND) ^ (XOR)"""

    def setUp(self):
        self.evset_1 = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [5, 9, 12, 3]},
        )
        self.evset_2 = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [7, 10, 6, 15]},
            same_sampling_as=self.evset_1,
        )

    def test_correct_and(self) -> None:
        """Test correct AND operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"bitwise_and_x_x": [5, 8, 4, 3]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(
            self,
            bitwise_and(self.evset_1, self.evset_2),
            expected
        )

    def test_correct_or(self) -> None:
        """Test correct OR operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"bitwise_or_x_x": [7, 11, 14, 15]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(
            self,
            bitwise_or(self.evset_1, self.evset_2),
            expected
        )

    def test_correct_xor(self) -> None:
        """Test correct XOR operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"bitwise_xor_x_x": [2, 3, 10, 12]},
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(
            self,
            bitwise_xor(self.evset_1, self.evset_2),
            expected
        )


if __name__ == "__main__":
    absltest.main()
