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

from temporian.core.operators.scalar import (
    AddScalarOperator,
    SubtractScalarOperator,
    MultiplyScalarOperator,
    DivideScalarOperator,
    FloorDivScalarOperator,
    EqualScalarOperator,
    GreaterScalarOperator,
    LessScalarOperator,
)
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.scalar import (
    AddScalarNumpyImplementation,
    SubtractScalarNumpyImplementation,
    MultiplyScalarNumpyImplementation,
    DivideScalarNumpyImplementation,
    FloorDivideScalarNumpyImplementation,
    EqualScalarNumpyImplementation,
    GreaterScalarNumpyImplementation,
    LessScalarNumpyImplementation,
)


class ArithmeticScalarNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators:
    addition, subtraction, division and multiplication"""

    def setUp(self):
        self.evset = EventSet.from_dataframe(
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

        self.node = self.evset.node()

    def test_correct_add(self) -> None:
        """Test correct sum operator."""

        value = 10.0

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 20.0],
                    [0, 2.0, 10.0],
                    [0, 3.0, 22.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 40.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = AddScalarOperator(
            input=self.node,
            value=value,
        )

        impl = AddScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_subtraction(self) -> None:
        """Test correct subtraction operator."""

        value = 10.0

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 0.0],
                    [0, 2.0, -10.0],
                    [0, 3.0, 2.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 20.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = SubtractScalarOperator(
            input=self.node,
            value=value,
        )

        impl = SubtractScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_subtraction_value_first(self) -> None:
        """Test correct subtraction operator when value is the first.
        operand.
        """
        value = 10.0

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 0.0],
                    [0, 2.0, 10.0],
                    [0, 3.0, -2.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, -20.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = SubtractScalarOperator(
            input=self.node,
            value=value,
            is_value_first=True,
        )

        impl = SubtractScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_multiplication(self) -> None:
        """Test correct multiplication operator."""

        value = 10.0

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 100.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, 120.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 300.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = MultiplyScalarOperator(
            input=self.node,
            value=value,
        )

        impl = MultiplyScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_division(self) -> None:
        """Test correct division operator."""

        value = 10.0

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 1.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, 1.2],
                    [0, 4.0, np.nan],
                    [0, 5.0, 3.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = DivideScalarOperator(
            input=self.node,
            value=value,
        )

        impl = DivideScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_division_with_value_as_numerator(self) -> None:
        """Test correct division operator with value as numerator."""
        value = 10.0

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 1.0],
                    [0, 2.0, np.inf],
                    [0, 3.0, 10.0 / 12.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 1.0 / 3.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = DivideScalarOperator(
            input=self.node,
            value=value,
            is_value_first=True,
        )

        impl = DivideScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_floor_division(self) -> None:
        """Test correct floor division operator."""

        value = 10.0

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 1.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, 1.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 3.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = FloorDivScalarOperator(
            input=self.node,
            value=value,
        )

        impl = FloorDivideScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_floor_division_with_value_as_numerator(self) -> None:
        """Test correct floor division operator with value as numerator."""
        value = 10.0

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 1.0],
                    [0, 2.0, np.inf],
                    [0, 3.0, 10.0 // 12.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 1.0 // 3.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = FloorDivScalarOperator(
            input=self.node,
            value=value,
            is_value_first=True,
        )

        impl = FloorDivideScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_sum_multi_index(self) -> None:
        """Test correct sum operator with multiple indexes."""

        evset = EventSet.from_dataframe(
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

        output_evset = EventSet.from_dataframe(
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
                    "sales",
                ],
            ),
            index_names=["store_id", "product_id"],
        )

        node = evset.node()

        operator = AddScalarOperator(
            input=node,
            value=value,
        )

        impl = AddScalarNumpyImplementation(operator)

        operator_output = impl.call(input=evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_addition_upcast(self) -> None:
        """Test correct addition operator with a value that would require
        an upcast in the feature dtype."""

        # Value: float, feature: int -> output should not be upcasted to float
        value = 10.0

        event_data = EventSet.from_dataframe(
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

        node = event_data.node()

        with self.assertRaises(ValueError):
            operator = AddScalarOperator(
                input=node,
                value=value,
            )

    def test_addition_with_string_value(self) -> None:
        """Test correct addition operator with string value."""

        value = "10"

        with self.assertRaises(ValueError):
            operator = AddScalarOperator(
                input=self.node,
                value=value,
            )

    def test_addition_with_int(self) -> None:
        """Test correct addition operator with int value."""

        evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10, 5.5],
                    [0, 2.0, 0, 3.0],
                    [0, 3.0, 12, 2.1],
                    [0, 4.0, -10, 3.3],
                    [0, 5.0, 30, 9],
                ],
                columns=["store_id", "timestamp", "sales", "revenue"],
            ),
            index_names=["store_id"],
        )

        node = evset.node()

        value = 10

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 20, 15.5],
                    [0, 2.0, 10, 13.0],
                    [0, 3.0, 22, 12.1],
                    [0, 4.0, 0, 13.3],
                    [0, 5.0, 40, 19.0],
                ],
                columns=[
                    "store_id",
                    "timestamp",
                    "sales",
                    "revenue",
                ],
            ),
            index_names=["store_id"],
        )

        operator = AddScalarOperator(
            input=node,
            value=value,
        )

        impl = AddScalarNumpyImplementation(operator)

        operator_output = impl.call(input=evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_equal(self) -> None:
        """Test correct equal operator."""

        self.evset = EventSet.from_dataframe(
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

        self.node = self.evset.node()

        value = 0.0

        output_evset = EventSet.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, False],
                    [0, 2.0, True],
                    [0, 3.0, False],
                    [0, 4.0, False],
                    [0, 5.0, True],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = EqualScalarOperator(
            input=self.node,
            value=value,
        )

        impl = EqualScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_greater_scalar(self) -> None:
        event_data = EventSet.from_dataframe(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, 2, 3]})
        )
        expected_data = EventSet.from_dataframe(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [False, False, True]})
        )

        event = event_data.node()
        operator = GreaterScalarOperator(
            input=event,
            value=2,
        )
        impl = GreaterScalarNumpyImplementation(operator)
        operator_output = impl.call(input=event_data)
        self.assertEqual(expected_data, operator_output["output"])

    def test_less_scalar(self) -> None:
        event_data = EventSet.from_dataframe(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, 2, 3]})
        )
        expected_data = EventSet.from_dataframe(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [True, False, False]})
        )

        event = event_data.node()
        operator = LessScalarOperator(
            input=event,
            value=2,
        )
        impl = LessScalarNumpyImplementation(operator)
        operator_output = impl.call(input=event_data)

        self.assertEqual(expected_data, operator_output["output"])


if __name__ == "__main__":
    absltest.main()
