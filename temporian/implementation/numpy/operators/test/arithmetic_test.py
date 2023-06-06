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

from temporian.core.operators.binary import (
    AddOperator,
    SubtractOperator,
    MultiplyOperator,
    DivideOperator,
    FloorDivOperator,
    ModuloOperator,
    PowerOperator,
)
from temporian.implementation.numpy.data.io import pd_dataframe_to_event_set
from temporian.implementation.numpy.operators.binary import (
    AddNumpyImplementation,
    SubtractNumpyImplementation,
    MultiplyNumpyImplementation,
    DivideNumpyImplementation,
    FloorDivNumpyImplementation,
    ModuloNumpyImplementation,
    PowerNumpyImplementation,
)


class ArithmeticNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators:
    addition, subtraction, division and multiplication"""

    def setUp(self):
        self.evset_1 = pd_dataframe_to_event_set(
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
        self.evset_2 = pd_dataframe_to_event_set(
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
            same_sampling_as=self.evset_1,
        )
        self.evset_3 = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 1.0, 0.0],
                    [0, 2.0, 20.0],
                    [0, 3.0, np.nan],
                    [0, 4.0, 0.0],
                    [0, 5.0, 7.0],
                ],
                columns=["store_id", "timestamp", "costs"],
            ),
            index_names=["store_id"],
            same_sampling_as=self.evset_1,
        )
        self.node_1 = self.evset_1.node()
        self.node_2 = self.evset_2.node()
        self.node_3 = self.evset_3.node()

    def test_correct_sum(self) -> None:
        """Test correct sum operator."""

        output_evset = pd_dataframe_to_event_set(
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
        operator.outputs["output"].check_same_sampling(self.node_1)

        sum_implementation = AddNumpyImplementation(operator)

        operator_output = sum_implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )
        self.assertTrue(output_evset == operator_output["output"])

    def test_correct_subtraction(self) -> None:
        """Test correct subtraction operator."""

        output_evset = pd_dataframe_to_event_set(
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

        output_evset = pd_dataframe_to_event_set(
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

        output_evset = pd_dataframe_to_event_set(
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

    def test_correct_floordiv(self) -> None:
        """Test correct floor division operator."""

        # Using evset_1 and evset_3
        output_evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 1.0, np.inf],
                    [0, 2.0, 0.0],
                    [0, 3.0, np.nan],
                    [0, 4.0, np.nan],
                    [0, 5.0, 4.0],
                ],
                columns=["store_id", "timestamp", "floordiv_sales_costs"],
            ),
            index_names=["store_id"],
        )

        operator = FloorDivOperator(
            input_1=self.node_1,
            input_2=self.node_3,
        )

        div_implementation = FloorDivNumpyImplementation(operator)

        operator_output = div_implementation.call(
            input_1=self.evset_1, input_2=self.evset_3
        )

        self.assertTrue(output_evset == operator_output["output"])

    def test_correct_modulo(self) -> None:
        """Test correct modulo operator."""

        # Using evset_1 and evset_3
        output_evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 1.0, np.nan],
                    [0, 2.0, 0.0],
                    [0, 3.0, np.nan],
                    [0, 4.0, np.nan],
                    [0, 5.0, 2.0],
                ],
                columns=["store_id", "timestamp", "mod_sales_costs"],
            ),
            index_names=["store_id"],
        )

        operator = ModuloOperator(
            input_1=self.node_1,
            input_2=self.node_3,
        )

        op_implementation = ModuloNumpyImplementation(operator)

        operator_output = op_implementation.call(
            input_1=self.evset_1, input_2=self.evset_3
        )

        self.assertTrue(output_evset == operator_output["output"])

    def test_correct_power(self) -> None:
        """Test correct power operator."""

        # Using evset_1 and evset_3
        output_evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 1.0, 1.0],
                    [0, 2.0, 0.0],
                    [0, 3.0, np.nan],
                    [0, 4.0, 1.0],
                    [0, 5.0, 30**7],
                ],
                columns=["store_id", "timestamp", "pow_sales_costs"],
            ),
            index_names=["store_id"],
        )

        operator = PowerOperator(
            input_1=self.node_1,
            input_2=self.node_3,
        )

        op_implementation = PowerNumpyImplementation(operator)

        operator_output = op_implementation.call(
            input_1=self.evset_1, input_2=self.evset_3
        )

        self.assertTrue(output_evset == operator_output["output"])


if __name__ == "__main__":
    absltest.main()
