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

from temporian.test.utils import f64, assertOperatorResult


class ArithmeticScalarTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators:
    addition, subtraction, division and multiplication"""

    def setUp(self):
        self.evset = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([10, 0, 12, np.nan, 30])},
        )

    def test_correct_add(self) -> None:
        """Test correct sum operator."""
        value = 10.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([20, 10, 22, np.nan, 40])},
            same_sampling_as=self.evset,
        )
        # evset first (value right)
        assertOperatorResult(self, self.evset + value, expected)
        # value first
        assertOperatorResult(self, value + self.evset, expected)

    def test_correct_subtraction(self) -> None:
        """Test correct subtraction operator."""
        value = 10.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([0, -10, 2, np.nan, 20])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset - value, expected)

    def test_correct_subtraction_value_first(self) -> None:
        """Test correct subtraction operator when value is the first.
        operand.
        """
        value = 10.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([0, 10, -2, np.nan, -20])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, value - self.evset, expected)

    def test_correct_multiplication(self) -> None:
        """Test correct multiplication operator."""
        value = 10.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([100, 0, 120, np.nan, 300])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset * value, expected)

    def test_correct_division(self) -> None:
        """Test correct division operator."""
        value = 10.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([1, 0, 1.2, np.nan, 3])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset / value, expected)

    def test_correct_division_with_value_as_numerator(self) -> None:
        """Test correct division operator with value as numerator."""
        value = 10.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([1, np.inf, 10 / 12, np.nan, 1 / 3])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, value / self.evset, expected)

    def test_correct_floor_division(self) -> None:
        """Test correct floor division operator."""
        value = 10.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([1, 0, 1, np.nan, 3])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset // value, expected)

    def test_correct_floor_division_with_value_as_numerator(self) -> None:
        """Test correct floor division operator with value as numerator."""
        value = 10.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([1, np.inf, 10 // 12, np.nan, 1 // 3])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, value // self.evset, expected)

    def test_correct_modulo(self) -> None:
        """Test correct modulo operator."""
        value = 7.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([3, 0, 5, np.nan, 2])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset % value, expected)

    def test_correct_modulo_value_first(self) -> None:
        """Test correct modulo operator with value to the left."""
        value = 25.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([5, np.nan, 1, np.nan, 25])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, value % self.evset, expected)

    def test_correct_power(self) -> None:
        """Test correct power operator."""
        value = 2.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([100, 0, 144, np.nan, 900])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset ** value, expected)

    def test_correct_power_value_first(self) -> None:
        """Test correct power operator with value as base."""
        value = 2.0
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([1024, 1, 4096, np.nan, 2 ** 30])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, value ** self.evset, expected)

    def test_correct_sum_multi_index(self) -> None:
        """Test correct sum operator with multiple indexes."""

        evset_index = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 2],
                "product_id": [101, 203, 83, 21, 310],
                "sales": [10, 0, 12, np.nan, 30],
            },
            indexes=["store_id", "product_id"],
        )
        value = 10.0

        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 2],
                "product_id": [101, 203, 83, 21, 310],
                "sales": [20, 10, 22, np.nan, 40],
            },
            indexes=["store_id", "product_id"],
            same_sampling_as=evset_index,
        )

        # evset first
        assertOperatorResult(self, evset_index + value, expected)
        # value first
        assertOperatorResult(self, value + evset_index, expected)

    def test_addition_upcast(self) -> None:
        """Test correct addition operator with a value that would require
        an upcast in the feature dtype."""

        # Value: float, feature: int -> output should not be upcasted to float
        value = 10.0

        evset = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": [10, 0, 12, -10, 30]},  # int64 feature
            same_sampling_as=self.evset,
        )
        with self.assertRaisesRegex(ValueError, "Use cast()"):
            _ = evset + value

    def test_addition_with_string_value(self) -> None:
        """Test correct addition operator with string value."""

        value = "10"

        with self.assertRaises(ValueError):
            _ = self.evset + value

    def test_addition_with_int(self) -> None:
        """Test correct addition operator with int value."""
        value = 10
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"sales": f64([20, 10, 22, np.nan, 40])},
            same_sampling_as=self.evset,
        )
        # evset first (value right)
        assertOperatorResult(self, self.evset + value, expected)
        # value first
        assertOperatorResult(self, value + self.evset, expected)


if __name__ == "__main__":
    absltest.main()
