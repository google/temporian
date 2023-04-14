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
import pandas as pd

from temporian.core.operators.boolean.equal_scalar import EqualScalarOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.boolean import equal


class EqualScalarOperatorTest(absltest.TestCase):
    """Equal scalar operator test."""

    def test_float_equal(self) -> None:
        """Test equal when value is a float."""
        df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
                [3.0, 12.0],
                [4.0, 13.0],
                [5.0, 14.0],
                [6.0, 15.0],
            ],
            columns=["timestamp", "sales"],
        )

        input_event_data = NumpyEvent.from_dataframe(df)

        input_event = input_event_data.schema()

        new_df = pd.DataFrame(
            [
                [1.0, True],
                [2.0, False],
                [3.0, False],
                [4.0, False],
                [5.0, False],
                [6.0, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_10.0",
            ],
        )

        operator = EqualScalarOperator(event_1=input_event, value=10.0)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event_1=input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_string_equal(self) -> None:
        """Test equal when value is a string."""
        df = pd.DataFrame(
            [
                [1.0, "A"],
                [2.0, "A"],
                [3.0, "B"],
                [4.0, "B"],
                [5.0, "10.0"],
                [6.0, "10"],
            ],
            columns=["timestamp", "costs"],
        )

        input_event_data = NumpyEvent.from_dataframe(df)

        input_event = input_event_data.schema()

        new_df = pd.DataFrame(
            [
                [1.0, False],
                [2.0, False],
                [3.0, False],
                [4.0, False],
                [5.0, False],
                [6.0, True],
            ],
            columns=[
                "timestamp",
                "costs_equal_10",
            ],
        )

        operator = EqualScalarOperator(event_1=input_event, value="10")
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event_1=input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_int_equal(self) -> None:
        """Test equal when value is an int."""

        df = pd.DataFrame(
            [
                [1.0, 10],
                [2.0, 11],
                [3.0, 100],
                [4.0, 20],
                [5.0, 0],
                [6.0, 40],
            ],
            columns=["timestamp", "weather"],
        )

        input_event_data = NumpyEvent.from_dataframe(df)

        input_event = input_event_data.schema()

        new_df = pd.DataFrame(
            [
                [1.0, True],
                [2.0, False],
                [3.0, False],
                [4.0, False],
                [5.0, False],
                [6.0, False],
            ],
            columns=[
                "timestamp",
                "weather_equal_10",
            ],
        )

        operator = EqualScalarOperator(event_1=input_event, value=10)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event_1=input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_int_equal_with_int32_column(self) -> None:
        """Test equal when value is an int64 and a column is an int32."""

        df = pd.DataFrame(
            [
                [1.0, 10],
                [2.0, 11],
                [3.0, 100],
                [4.0, 20],
                [5.0, 0],
                [6.0, 40],
            ],
            columns=["timestamp", "weather"],
        )

        df["weather"] = df["weather"].astype(np.int32)

        input_event_data = NumpyEvent.from_dataframe(df)

        input_event = input_event_data.schema()

        new_df = pd.DataFrame(
            [
                [1.0, True],
                [2.0, False],
                [3.0, False],
                [4.0, False],
                [5.0, False],
                [6.0, False],
            ],
            columns=[
                "timestamp",
                "weather_equal_10",
            ],
        )

        # The column is int32, but value is int64. It should still work.
        operator = EqualScalarOperator(event_1=input_event, value=10)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event_1=input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_float_equal_with_float32_column(self) -> None:
        """Test equal when value is an int64 and a column is an int32."""

        df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
                [3.0, 12.0],
                [4.0, 13.0],
                [5.0, 14.0],
                [6.0, 15.0],
            ],
            columns=["timestamp", "sales"],
        )

        df["sales"] = df["sales"].astype(np.float32)

        input_event_data = NumpyEvent.from_dataframe(df)

        input_event = input_event_data.schema()

        new_df = pd.DataFrame(
            [
                [1.0, True],
                [2.0, False],
                [3.0, False],
                [4.0, False],
                [5.0, False],
                [6.0, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_10.0",
            ],
        )

        # The column is float32, but value is float64. It should still work.
        operator = EqualScalarOperator(event_1=input_event, value=10.0)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event_1=input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_value_unsupported_type(self):
        """Test operator raises error when value is not a supported type."""
        df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
                [3.0, 12.0],
                [4.0, 13.0],
                [5.0, 14.0],
                [6.0, 15.0],
            ],
            columns=["timestamp", "sales"],
        )

        input_event_data = NumpyEvent.from_dataframe(df)

        input_event = input_event_data.schema()

        class MyClass:
            def __init__(self, a_variable) -> None:
                self.a_variable = a_variable

        value = MyClass(10)

        with self.assertRaises(ValueError):
            operator = EqualScalarOperator(event_1=input_event, value=value)


if __name__ == "__main__":
    absltest.main()
