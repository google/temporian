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
from temporian.core.data.node import Node, Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.binary import (
    AddOperator,
    SubtractOperator,
    MultiplyOperator,
    DivideOperator,
    EqualOperator,
)
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.binary import (
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
        self.evset_1 = EventSet.from_dataframe(
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
        self.evset_2 = EventSet.from_dataframe(
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
        # set same sampling
        for index_key, index_data in self.evset_1.data.items():
            self.evset_2[index_key].timestamps = index_data.timestamps
        self.node_1 = self.evset_1.node()
        self.node_2 = self.evset_2.node()

        self.sampling = Sampling(
            [("store_id", DType.INT64)], is_unix_timestamp=False
        )
        self.node_1 = Node(
            [Feature("sales", DType.FLOAT64)],
            sampling=self.sampling,
            creator=None,
        )
        self.node_2 = Node(
            [Feature("costs", DType.FLOAT64)],
            sampling=self.sampling,
            creator=None,
        )

    def test_correct_sum(self) -> None:
        """Test correct sum operator."""

        output_evset = EventSet.from_dataframe(
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
            input_1=self.node_1,
            input_2=self.node_2,
        )

        sum_implementation = AddNumpyImplementation(operator)

        operator_output = sum_implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )
        self.assertTrue(output_evset == operator_output["output"])

    def test_correct_subtraction(self) -> None:
        """Test correct subtraction operator."""

        output_evset = EventSet.from_dataframe(
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
            input_1=self.node_1,
            input_2=self.node_2,
        )

        sub_implementation = SubtractNumpyImplementation(operator)
        operator_output = sub_implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )
        self.assertTrue(output_evset == operator_output["output"])

    def test_correct_multiplication(self) -> None:
        """Test correct multiplication operator."""

        output_evset = EventSet.from_dataframe(
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
            input_1=self.node_1,
            input_2=self.node_2,
        )

        mult_implementation = MultiplyNumpyImplementation(operator)

        operator_output = mult_implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )
        self.assertTrue(output_evset == operator_output["output"])

    def test_correct_division(self) -> None:
        """Test correct division operator."""

        output_evset = EventSet.from_dataframe(
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
            input_1=self.node_1,
            input_2=self.node_2,
        )

        div_implementation = DivideNumpyImplementation(operator)

        operator_output = div_implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )

        self.assertTrue(output_evset == operator_output["output"])

    def test_correct_equal(self) -> None:
        """Test correct equal operator."""
        self.evset_1 = EventSet.from_dataframe(
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

        self.evset_2 = EventSet.from_dataframe(
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
        for index_key, index_data in self.evset_1.iterindex():
            self.evset_2[index_key].timestamps = index_data.timestamps

        self.node_1 = self.evset_1.node()
        self.node_2 = self.evset_2.node()
        self.node_1._sampling = self.node_2._sampling

        output_evset = EventSet.from_dataframe(
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
            input_1=self.node_1,
            input_2=self.node_2,
        )

        equal_implementation = EqualNumpyImplementation(operator)

        operator_output = equal_implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )

        self.assertEqual(output_evset, operator_output["output"])


if __name__ == "__main__":
    absltest.main()
