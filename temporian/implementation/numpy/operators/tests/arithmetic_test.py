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
import numpy as np

from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.arithmetic import ArithmeticOperation
from temporian.core.operators.arithmetic import ArithmeticOperator
from temporian.core.operators.arithmetic import Resolution
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy.operators import arithmetic


class ArithmeticOperatorTest(absltest.TestCase):
    """Test ArithmeticOperator."""

    def setUp(self):
        self.numpy_input_sampling = NumpySampling(
            index=["store_id"],
            data={("A",): np.array([1, 2, 3, 4, 5])},
        )

        self.numpy_event_1 = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([10.0, 0.0, 12.0, np.nan, 30.0]),
                    ),
                ],
            },
            sampling=self.numpy_input_sampling,
        )

        self.numpy_event_2 = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([0, 20.0, np.nan, 6.0, 10.0]),
                    ),
                ],
            },
            sampling=self.numpy_input_sampling,
        )

        self.sampling = Sampling(["store_id"])
        self.event_1 = Event(
            [Feature("sales", float)],
            sampling=self.sampling,
            creator=None,
        )
        self.event_2 = Event(
            [Feature("costs", float)],
            sampling=self.sampling,
            creator=None,
        )

    def test_correct_sum(self) -> None:
        """Test correct sum operator."""
        # DATA

        numpy_output_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="add_sales_costs",
                        data=np.array([10.0, 20.0, np.nan, np.nan, 40.0]),
                    ),
                ],
            },
            sampling=self.numpy_input_sampling,
        )

        operator = ArithmeticOperator(
            event_1=self.event_1,
            event_2=self.event_2,
            operation=ArithmeticOperation.ADDITION,
            resolution=Resolution.PER_FEATURE_IDX,
        )

        sum_implementation = arithmetic.ArithmeticNumpyImplementation(operator)

        operator_output = sum_implementation(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertEqual(
            True,
            numpy_output_event == operator_output["event"],
        )

    def test_correct_substraction(self) -> None:
        """Test correct substraction operator."""

        numpy_output_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="sub_sales_costs",
                        data=np.array([10.0, -20.0, np.nan, np.nan, 20.0]),
                    ),
                ],
            },
            sampling=self.numpy_input_sampling,
        )

        operator = ArithmeticOperator(
            event_1=self.event_1,
            event_2=self.event_2,
            operation=ArithmeticOperation.SUBTRACTION,
            resolution=Resolution.PER_FEATURE_IDX,
        )

        sub_implementation = arithmetic.ArithmeticNumpyImplementation(operator)

        operator_output = sub_implementation(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertEqual(
            True,
            numpy_output_event == operator_output["event"],
        )

    def test_correct_multiplication(self) -> None:
        """Test correct multiplication operator."""

        numpy_output_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="mult_sales_costs",
                        data=np.array([0.0, 0.0, np.nan, np.nan, 300.0]),
                    ),
                ],
            },
            sampling=self.numpy_input_sampling,
        )

        operator = ArithmeticOperator(
            event_1=self.event_1,
            event_2=self.event_2,
            operation=ArithmeticOperation.MULTIPLICATION,
            resolution=Resolution.PER_FEATURE_IDX,
        )

        mult_implementation = arithmetic.ArithmeticNumpyImplementation(operator)

        operator_output = mult_implementation(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertEqual(
            True,
            numpy_output_event == operator_output["event"],
        )

    def test_correct_division(self) -> None:
        """Test correct division operator."""

        numpy_output_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="div_sales_costs",
                        data=np.array([np.inf, 0.0, np.nan, np.nan, 3.0]),
                    ),
                ],
            },
            sampling=self.numpy_input_sampling,
        )

        operator = ArithmeticOperator(
            event_1=self.event_1,
            event_2=self.event_2,
            operation=ArithmeticOperation.DIVISION,
            resolution=Resolution.PER_FEATURE_IDX,
        )

        div_implementation = arithmetic.ArithmeticNumpyImplementation(operator)

        operator_output = div_implementation(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertEqual(
            True,
            numpy_output_event == operator_output["event"],
        )


if __name__ == "__main__":
    absltest.main()
