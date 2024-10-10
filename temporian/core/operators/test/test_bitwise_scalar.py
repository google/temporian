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

import numpy as np
from absl.testing import absltest
from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult
from temporian.core.operators.scalar import (
    bitwise_and_scalar,
    bitwise_or_scalar,
    bitwise_xor_scalar,
    left_shift_scalar,
    right_shift_scalar,
)


class ArithmeticScalarTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators:
    addition, subtraction, division and multiplication"""

    def setUp(self):
        self.evset = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [2, 5, 7, 9]},
        )

    def test_correct_and(self) -> None:
        """Test correct and operator."""
        value = 3
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [2, 1, 3, 1]},
            same_sampling_as=self.evset,
        )

        assertOperatorResult(
            self,
            bitwise_and_scalar(self.evset, value),
            expected
        )

    def test_correct_or(self) -> None:
        """Test correct or operator."""
        value = 3
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [3, 7, 7, 11]},
            same_sampling_as=self.evset,
        )

        assertOperatorResult(
            self,
            bitwise_or_scalar(self.evset, value),
            expected
        )

    def test_correct_xor(self) -> None:
        """Test correct xor operator."""
        value = 3
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [1, 6, 4, 10]},
            same_sampling_as=self.evset,
        )

        assertOperatorResult(
            self,
            bitwise_xor_scalar(self.evset, value),
            expected
        )

    def test_correct_left_shift(self) -> None:
        """Test correct left shift operator."""
        value = 2
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [8, 20, 28, 36]},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(
            self, left_shift_scalar(self.evset, value), expected
        )

    def test_correct_right_shift(self) -> None:
        """Test correct right shift operator."""
        value = 1
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [1, 2, 3, 4]},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(
            self, right_shift_scalar(self.evset, value), expected
        )


if __name__ == "__main__":
    absltest.main()
