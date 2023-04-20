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
import pandas as pd
from absl.testing import absltest

from temporian.core.operators.arithmetic_scalar import (
    AddScalarOperator,
    SubtractScalarOperator,
    MultiplyScalarOperator,
    DivideScalarOperator,
    FloorDivScalarOperator,
    EqualScalarOperator,
    NegateOperator,
)
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.arithmetic_scalar import (
    AddScalarNumpyImplementation,
    SubtractScalarNumpyImplementation,
    MultiplyScalarNumpyImplementation,
    DivideScalarNumpyImplementation,
    FloorDivideScalarNumpyImplementation,
    EqualScalarNumpyImplementation,
    NegateNumpyImplementation,
)


class ArithmeticScalarNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators:
    addition, subtraction, division and multiplication"""

    def setUp(self):
        self.event_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, 12.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 30.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        self.event = self.event_data.schema()

    def test_correct_add(self) -> None:
        """Test correct sum operator."""

        value = 10.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 20.0],
                    [0, 2.0, 10.0],
                    [0, 3.0, 22.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 40.0],
                ],
                columns=["store_id", "timestamp", "add_sales_10.0"],
            ),
            index_names=["store_id"],
        )

        operator = AddScalarOperator(
            event=self.event,
            value=value,
        )

        impl = AddScalarNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_substraction(self) -> None:
        """Test correct substraction operator."""

        value = 10.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 0.0],
                    [0, 2.0, -10.0],
                    [0, 3.0, 2.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 20.0],
                ],
                columns=["store_id", "timestamp", "sub_sales_10.0"],
            ),
            index_names=["store_id"],
        )

        operator = SubtractScalarOperator(
            event=self.event,
            value=value,
        )

        impl = SubtractScalarNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_substraction_value_first(self) -> None:
        """Test correct substraction operator when value is the first.
        operand.
        """
        value = 10.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 0.0],
                    [0, 2.0, 10.0],
                    [0, 3.0, -2.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, -20.0],
                ],
                columns=["store_id", "timestamp", "sub_sales_10.0"],
            ),
            index_names=["store_id"],
        )

        operator = SubtractScalarOperator(
            event=self.event,
            value=value,
            is_value_first=True,
        )

        impl = SubtractScalarNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_multiplication(self) -> None:
        """Test correct multiplication operator."""

        value = 10.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 100.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, 120.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 300.0],
                ],
                columns=["store_id", "timestamp", "mult_sales_10.0"],
            ),
            index_names=["store_id"],
        )

        operator = MultiplyScalarOperator(
            event=self.event,
            value=value,
        )

        impl = MultiplyScalarNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_division(self) -> None:
        """Test correct division operator."""

        value = 10.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 1.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, 1.2],
                    [0, 4.0, np.nan],
                    [0, 5.0, 3.0],
                ],
                columns=["store_id", "timestamp", "div_sales_10.0"],
            ),
            index_names=["store_id"],
        )

        operator = DivideScalarOperator(
            event=self.event,
            value=value,
        )

        impl = DivideScalarNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_division_with_value_as_numerator(self) -> None:
        """Test correct division operator with value as numerator."""
        value = 10.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 1.0],
                    [0, 2.0, np.inf],
                    [0, 3.0, 10.0 / 12.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 1.0 / 3.0],
                ],
                columns=["store_id", "timestamp", "div_sales_10.0"],
            ),
            index_names=["store_id"],
        )

        operator = DivideScalarOperator(
            event=self.event,
            value=value,
            is_value_first=True,
        )

        impl = DivideScalarNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_floor_division(self) -> None:
        """Test correct floor division operator."""

        value = 10.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 1.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, 1.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 3.0],
                ],
                columns=["store_id", "timestamp", "floordiv_sales_10.0"],
            ),
            index_names=["store_id"],
        )

        operator = FloorDivScalarOperator(
            event=self.event,
            value=value,
        )

        impl = FloorDivideScalarNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_floor_division_with_value_as_numerator(self) -> None:
        """Test correct floor division operator with value as numerator."""
        value = 10.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 1.0],
                    [0, 2.0, np.inf],
                    [0, 3.0, 10.0 // 12.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 1.0 // 3.0],
                ],
                columns=["store_id", "timestamp", "floordiv_sales_10.0"],
            ),
            index_names=["store_id"],
        )

        operator = FloorDivScalarOperator(
            event=self.event,
            value=value,
            is_value_first=True,
        )

        impl = FloorDivideScalarNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_negation(self) -> None:
        """Test correct negation operator."""

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, -10.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, -12.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, -30.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = NegateOperator(event=self.event)

        impl = NegateNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_sum_multi_index(self) -> None:
        """Test correct sum operator with multiple indexes."""

        event_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 101, 1.0, 10.0],
                    [0, 203, 2.0, 0.0],
                    [1, 83, 3.0, 12.0],
                    [1, 21, 4.0, np.nan],
                    [2, 310, 5.0, 30.0],
                ],
                columns=["store_id", "product_id", "timestamp", "sales"],
            ),
            index_names=["store_id", "product_id"],
        )

        value = 10.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 101, 1.0, 20.0],
                    [0, 203, 2.0, 10.0],
                    [1, 83, 3.0, 22.0],
                    [1, 21, 4.0, np.nan],
                    [2, 310, 5.0, 40.0],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "add_sales_10.0",
                ],
            ),
            index_names=["store_id", "product_id"],
        )

        event = event_data.schema()

        operator = AddScalarOperator(
            event=event,
            value=value,
        )

        impl = AddScalarNumpyImplementation(operator)

        operator_output = impl.call(event=event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_addition_different_dtypes(self) -> None:
        """Test correct addition operator with different dtypes."""

        value = 10

        with self.assertRaises(ValueError):
            operator = AddScalarOperator(
                event=self.event,
                value=value,
            )

    def test_addition_with_string_value(self) -> None:
        """Test correct addition operator with string value."""

        value = "10"

        with self.assertRaises(ValueError):
            operator = AddScalarOperator(
                event=self.event,
                value=value,
            )

    def test_addition_with_int(self) -> None:
        """Test correct addition operator with int value."""

        event_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10],
                    [0, 2.0, 0],
                    [0, 3.0, 12],
                    [0, 4.0, -10],
                    [0, 5.0, 30],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        event = event_data.schema()

        value = 10

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 20],
                    [0, 2.0, 10],
                    [0, 3.0, 22],
                    [0, 4.0, 0],
                    [0, 5.0, 40],
                ],
                columns=["store_id", "timestamp", "add_sales_10"],
            ),
            index_names=["store_id"],
        )

        operator = AddScalarOperator(
            event=event,
            value=value,
        )

        impl = AddScalarNumpyImplementation(operator)

        operator_output = impl.call(event=event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])

    def test_correct_equal(self) -> None:
        """Test correct equal operator."""

        self.event_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, 0.1],
                    [0, 4.0, np.nan],
                    [0, 5.0, 0.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        self.event = self.event_data.schema()

        value = 0.0

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, False],
                    [0, 2.0, True],
                    [0, 3.0, False],
                    [0, 4.0, False],
                    [0, 5.0, True],
                ],
                columns=["store_id", "timestamp", "equal_sales_0.0"],
            ),
            index_names=["store_id"],
        )

        operator = EqualScalarOperator(
            event=self.event,
            value=value,
        )

        impl = EqualScalarNumpyImplementation(operator)

        operator_output = impl.call(event=self.event_data)

        self.assertEqual(numpy_output_event, operator_output["event"])


if __name__ == "__main__":
    absltest.main()
