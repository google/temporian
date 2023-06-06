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
    EqualScalarOperator,
    NotEqualScalarOperator,
    GreaterScalarOperator,
    LessScalarOperator,
    GreaterEqualScalarOperator,
    LessEqualScalarOperator,
)
from temporian.implementation.numpy.operators.scalar import (
    EqualScalarNumpyImplementation,
    NotEqualScalarNumpyImplementation,
    GreaterScalarNumpyImplementation,
    LessScalarNumpyImplementation,
    GreaterEqualScalarNumpyImplementation,
    LessEqualScalarNumpyImplementation,
)
from temporian.implementation.numpy.data.io import pd_dataframe_to_event_set


class ArithmeticScalarNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators:
    addition, subtraction, division and multiplication"""

    def setUp(self):
        self.evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0],
                    [0, 2.0, 0.0],
                    [1, 3.0, 12.0],
                    [1, 4.0, np.nan],
                    [1, 5.0, 30.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        self.node = self.evset.node()

    def test_correct_equal(self) -> None:
        """Test correct equal operator."""
        value = 12.0

        output_evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 1.0, False],
                    [0, 2.0, False],
                    [1, 3.0, True],
                    [1, 4.0, False],
                    [1, 5.0, False],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        operator = EqualScalarOperator(
            input=self.node,
            value=value,
        )
        operator.outputs["output"].check_same_sampling(self.node)

        impl = EqualScalarNumpyImplementation(operator)

        operator_output = impl.call(input=self.evset)

        self.assertEqual(output_evset, operator_output["output"])

    def test_equal_nan(self) -> None:
        """Test equal operator against a nan value."""
        value = np.nan

        output_evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                [
                    [0, 1.0, False],
                    [0, 2.0, False],
                    [1, 3.0, False],
                    [1, 4.0, False],
                    [1, 5.0, False],
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
        event_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, 2, 3]})
        )
        expected_data = pd_dataframe_to_event_set(
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
        event_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, 2, 3]})
        )
        expected_data = pd_dataframe_to_event_set(
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

    def test_greater_equal_scalar(self) -> None:
        event_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, 2, 3]})
        )
        expected_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [False, True, True]})
        )

        event = event_data.node()
        operator = GreaterEqualScalarOperator(
            input=event,
            value=2,
        )
        impl = GreaterEqualScalarNumpyImplementation(operator)
        operator_output = impl.call(input=event_data)
        self.assertEqual(expected_data, operator_output["output"])

    def test_less_equal_scalar(self) -> None:
        event_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, 2, 3]})
        )
        expected_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [True, True, False]})
        )

        event = event_data.node()
        operator = LessEqualScalarOperator(
            input=event,
            value=2,
        )
        impl = LessEqualScalarNumpyImplementation(operator)
        operator_output = impl.call(input=event_data)
        self.assertEqual(expected_data, operator_output["output"])

    def test_not_equal_scalar(self) -> None:
        event_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, 2, 3]})
        )
        expected_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [True, False, True]})
        )

        event = event_data.node()
        operator = NotEqualScalarOperator(
            input=event,
            value=2,
        )
        impl = NotEqualScalarNumpyImplementation(operator)
        operator_output = impl.call(input=event_data)
        self.assertEqual(expected_data, operator_output["output"])

    def test_not_equal_nan(self) -> None:
        # a != nan should be True always (even if a=np.nan)
        event_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, 2, np.nan]})
        )
        expected_data = pd_dataframe_to_event_set(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [True, True, True]})
        )

        event = event_data.node()
        operator = NotEqualScalarOperator(
            input=event,
            value=np.nan,
        )
        impl = NotEqualScalarNumpyImplementation(operator)
        operator_output = impl.call(input=event_data)
        self.assertEqual(expected_data, operator_output["output"])


if __name__ == "__main__":
    absltest.main()
