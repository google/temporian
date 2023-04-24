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

from temporian.core.data.dtype import DType
from temporian.core.data.event import Event, Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.arithmetic import (
    AddOperator,
    SubtractOperator,
    MultiplyOperator,
    DivideOperator,
    EqualOperator,
)
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.arithmetic import (
    AddNumpyImplementation,
    SubtractNumpyImplementation,
    MultiplyNumpyImplementation,
    DivideNumpyImplementation,
    EqualNumpyImplementation,
)


class ArithmeticNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators:
    addition, subtraction, division and multiplication"""

    def setUp(self):
        self.numpy_event_1 = NumpyEvent.from_dataframe(
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

        self.numpy_event_2 = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 0.0],
                    [0, 2.0, 20.0],
                    [0, 3.0, np.nan],
                    [0, 4.0, 6.0],
                    [0, 5.0, 10.0],
                ],
                columns=["store_id", "timestamp", "costs"],
            ),
            index_names=["store_id"],
        )

        self.numpy_event_2.sampling = self.numpy_event_1.sampling

        self.sampling = Sampling([("store_id", DType.INT64)])
        self.event_1 = Event(
            [Feature("sales", DType.FLOAT64)],
            sampling=self.sampling,
            creator=None,
        )
        self.event_2 = Event(
            [Feature("costs", DType.FLOAT64)],
            sampling=self.sampling,
            creator=None,
        )

    def test_correct_sum(self) -> None:
        """Test correct sum operator."""

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0],
                    [0, 2.0, 20.0],
                    [0, 3.0, np.nan],
                    [0, 4.0, np.nan],
                    [0, 5.0, 40.0],
                ],
                columns=["store_id", "timestamp", "add_sales_costs"],
            ),
            index_names=["store_id"],
        )

        operator = AddOperator(
            event_1=self.event_1,
            event_2=self.event_2,
        )

        sum_implementation = AddNumpyImplementation(operator)

        operator_output = sum_implementation.call(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertTrue(numpy_output_event == operator_output["event"])

    def test_correct_subtraction(self) -> None:
        """Test correct subtraction operator."""

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0],
                    [0, 2.0, -20.0],
                    [0, 3.0, np.nan],
                    [0, 4.0, np.nan],
                    [0, 5.0, 20.0],
                ],
                columns=["store_id", "timestamp", "sub_sales_costs"],
            ),
            index_names=["store_id"],
        )

        operator = SubtractOperator(
            event_1=self.event_1,
            event_2=self.event_2,
        )

        sub_implementation = SubtractNumpyImplementation(operator)
        operator_output = sub_implementation.call(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )
        self.assertTrue(numpy_output_event == operator_output["event"])

    def test_correct_multiplication(self) -> None:
        """Test correct multiplication operator."""

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 0.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, np.nan],
                    [0, 4.0, np.nan],
                    [0, 5.0, 300.0],
                ],
                columns=["store_id", "timestamp", "mult_sales_costs"],
            ),
            index_names=["store_id"],
        )

        operator = MultiplyOperator(
            event_1=self.event_1,
            event_2=self.event_2,
        )

        mult_implementation = MultiplyNumpyImplementation(operator)

        operator_output = mult_implementation.call(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertTrue(numpy_output_event == operator_output["event"])

    def test_correct_division(self) -> None:
        """Test correct division operator."""

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, np.inf],
                    [0, 2.0, 0.0],
                    [0, 3.0, np.nan],
                    [0, 4.0, np.nan],
                    [0, 5.0, 3.0],
                ],
                columns=["store_id", "timestamp", "div_sales_costs"],
            ),
            index_names=["store_id"],
        )

        operator = DivideOperator(
            event_1=self.event_1,
            event_2=self.event_2,
        )

        div_implementation = DivideNumpyImplementation(operator)

        operator_output = div_implementation.call(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertTrue(numpy_output_event == operator_output["event"])

    def test_correct_equal(self) -> None:
        """Test correct equal operator."""
        self.numpy_event_1 = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0],
                    [0, 2.0, 1.0],
                    [0, 3.0, 12.0],
                    [0, 4.0, np.nan],
                    [0, 5.0, 30.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        self.numpy_event_2 = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0],
                    [0, 2.0, -1.0],
                    [0, 3.0, 12.000000001],
                    [0, 4.0, np.nan],
                    [0, 5.0, 30.000001],
                ],
                columns=["store_id", "timestamp", "costs"],
            ),
            index_names=["store_id"],
        )

        self.numpy_event_1.sampling = self.numpy_event_2.sampling
        self.event_1 = self.numpy_event_1.schema()
        self.event_2 = self.numpy_event_2.schema()
        self.event_1._sampling = self.event_2._sampling

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, True],
                    [0, 2.0, False],
                    [0, 3.0, False],
                    [0, 4.0, False],  # nan == nan is False
                    [0, 5.0, False],
                ],
                columns=["store_id", "timestamp", "equal_sales_costs"],
            ),
            index_names=["store_id"],
        )

        operator = EqualOperator(
            event_1=self.event_1,
            event_2=self.event_2,
        )

        equal_implementation = EqualNumpyImplementation(operator)

        operator_output = equal_implementation.call(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertEqual(numpy_output_event, operator_output["event"])


if __name__ == "__main__":
    absltest.main()
