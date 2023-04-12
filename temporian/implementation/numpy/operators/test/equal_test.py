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

from temporian.core.operators.equal import EqualOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators import equal


class EqualOperatorTest(absltest.TestCase):
    """Equal operator test."""

    def setUp(self):
        df = pd.DataFrame(
            [
                [1.0, 10.0, "A", 10],
                [2.0, np.nan, "A", 12],
                [3.0, 12.0, "B", 40],
                [4.0, 13.0, "B", 100],
                [5.0, 14.0, "10.0", 0],
                [6.0, 15.0, "10", 15],
            ],
            columns=["timestamp", "sales", "costs", "weather"],
        )

        self.input_event_data = NumpyEvent.from_dataframe(df)

        self.input_event = self.input_event_data.schema()

    def test_float_equal(self) -> None:
        """Test equal when value is a float."""
        new_df = pd.DataFrame(
            [
                [1.0, True, False, False],
                [2.0, False, False, False],
                [3.0, False, False, False],
                [4.0, False, False, False],
                [5.0, False, False, False],
                [6.0, False, False, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_10.0",
                "costs_equal_10.0",
                "weather_equal_10.0",
            ],
        )

        operator = EqualOperator(event=self.input_event, value=10.0)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event=self.input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_string_equal(self) -> None:
        """Test equal when value is a string."""
        new_df = pd.DataFrame(
            [
                [1.0, False, False, False],
                [2.0, False, False, False],
                [3.0, False, False, False],
                [4.0, False, False, False],
                [5.0, False, False, False],
                [6.0, False, True, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_10",
                "costs_equal_10",
                "weather_equal_10",
            ],
        )

        operator = EqualOperator(event=self.input_event, value="10")
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event=self.input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_int_equal(self) -> None:
        """Test equal when value is an int."""
        new_df = pd.DataFrame(
            [
                [1.0, False, False, True],
                [2.0, False, False, False],
                [3.0, False, False, False],
                [4.0, False, False, False],
                [5.0, False, False, False],
                [6.0, False, False, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_10",
                "costs_equal_10",
                "weather_equal_10",
            ],
        )

        operator = EqualOperator(event=self.input_event, value=10)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event=self.input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_int_equal_with_int32_column(self) -> None:
        """Test equal when value is an int64 and a column is an int32."""

        df = pd.DataFrame(
            [
                [1.0, 10.0, "A", 10],
                [2.0, np.nan, "A", 12],
                [3.0, 12.0, "B", 40],
                [4.0, 13.0, "B", 100],
                [5.0, 14.0, "10.0", 0],
                [6.0, 15.0, "10", 15],
            ],
            columns=["timestamp", "sales", "costs", "weather"],
        )

        # convert weather column to int32 to check different general dtypes.
        df["weather"] = df["weather"].astype(np.int32)

        self.input_event_data = NumpyEvent.from_dataframe(df)
        self.input_event = self.input_event_data.schema()

        new_df = pd.DataFrame(
            [
                [1.0, False, False, True],
                [2.0, False, False, False],
                [3.0, False, False, False],
                [4.0, False, False, False],
                [5.0, False, False, False],
                [6.0, False, False, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_10",
                "costs_equal_10",
                "weather_equal_10",
            ],
        )

        # default python int of value=10 is int64.
        # the comparison will be true because we check general dtypes.
        operator = EqualOperator(event=self.input_event, value=10)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event=self.input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_float_equal_with_float32_column(self) -> None:
        """Test equal when value is an int64 and a column is an int32."""

        df = pd.DataFrame(
            [
                [1.0, 10.0, "A", 10],
                [2.0, np.nan, "A", 12],
                [3.0, 12.0, "B", 40],
                [4.0, 13.0, "B", 100],
                [5.0, 14.0, "10.0", 0],
                [6.0, 15.0, "10", 15],
            ],
            columns=["timestamp", "sales", "costs", "weather"],
        )

        # convert weather column to int32 to check different general dtypes.
        df["sales"] = df["sales"].astype(np.float32)

        self.input_event_data = NumpyEvent.from_dataframe(df)
        self.input_event = self.input_event_data.schema()

        new_df = pd.DataFrame(
            [
                [1.0, True, False, False],
                [2.0, False, False, False],
                [3.0, False, False, False],
                [4.0, False, False, False],
                [5.0, False, False, False],
                [6.0, False, False, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_10.0",
                "costs_equal_10.0",
                "weather_equal_10.0",
            ],
        )

        # default python float of value=10 is float64.
        # the comparison will be true because we check general dtypes.
        operator = EqualOperator(event=self.input_event, value=10.0)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event=self.input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_boolean_equal(self) -> None:
        """Test equal when value is a boolean."""

        # this functionality is useless, but dont see the reason why raise an
        # error if the value is a boolean.
        df = pd.DataFrame(
            [
                [1.0, 10.0, "A", False],
                [2.0, np.nan, "A", False],
                [3.0, 12.0, "B", True],
                [4.0, 13.0, "B", True],
                [5.0, 14.0, "10.0", False],
                [6.0, 15.0, "10", False],
            ],
            columns=["timestamp", "sales", "costs", "weather"],
        )

        self.input_event_data = NumpyEvent.from_dataframe(df)
        self.input_event = self.input_event_data.schema()

        new_df = pd.DataFrame(
            [
                [1.0, False, False, False],
                [2.0, False, False, False],
                [3.0, False, False, True],
                [4.0, False, False, True],
                [5.0, False, False, False],
                [6.0, False, False, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_True",
                "costs_equal_True",
                "weather_equal_True",
            ],
        )

        operator = EqualOperator(event=self.input_event, value=True)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event=self.input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)


if __name__ == "__main__":
    absltest.main()
