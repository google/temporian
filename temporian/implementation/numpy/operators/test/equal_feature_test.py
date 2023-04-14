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

from temporian.core.operators.boolean.equal_feature import EqualFeatureOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.boolean import equal


class EqualFeatureOperatorTest(absltest.TestCase):
    """Equal feature operator test."""

    def test_float_equal(self) -> None:
        """Test equal when value is a float."""
        df_1 = pd.DataFrame(
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

        input_event_1_data = NumpyEvent.from_dataframe(df_1)

        input_event_1 = input_event_1_data.schema()

        df_2 = pd.DataFrame(
            [
                [1.0, 0],
                [2.0, np.nan],
                [3.0, 12.1],
                [4.0, 13.0],
                [5.0, 14.2],
                [6.0, 15.0],
            ],
            columns=["timestamp", "costs"],
        )

        input_event_2_data = NumpyEvent.from_dataframe(df_2)

        input_event_2 = input_event_2_data.schema()
        input_event_2.sampling = input_event_1.sampling

        new_df = pd.DataFrame(
            [
                [1.0, False],
                [2.0, False],  # np.nan comparison returns false
                [3.0, False],
                [4.0, True],
                [5.0, False],
                [6.0, True],
            ],
            columns=[
                "timestamp",
                "sales_equal_costs",
            ],
        )

        operator = EqualFeatureOperator(
            event_1=input_event_1, event_2=input_event_2
        )
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(
            event_1=input_event_1_data, event_2=input_event_2_data
        )["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        if not expected_event == (equal_event):
            print(expected_event.to_dataframe())
            print(equal_event.to_dataframe())

        self.assertEqual(equal_event, expected_event)

    # def test_string_equal(self) -> None:
    #     """Test equal when value is a string."""
    #     df = pd.DataFrame(
    #         [
    #             [1.0, "A"],
    #             [2.0, "A"],
    #             [3.0, "B"],
    #             [4.0, "B"],
    #             [5.0, "10.0"],
    #             [6.0, "10"],
    #         ],
    #         columns=["timestamp", "costs"],
    #     )

    #     input_event_data = NumpyEvent.from_dataframe(df)

    #     input_event = input_event_data.schema()

    #     new_df = pd.DataFrame(
    #         [
    #             [1.0, False],
    #             [2.0, False],
    #             [3.0, False],
    #             [4.0, False],
    #             [5.0, False],
    #             [6.0, True],
    #         ],
    #         columns=[
    #             "timestamp",
    #             "costs_equal_10",
    #         ],
    #     )

    #     operator = EqualScalarOperator(event=input_event, value="10")
    #     impl = equal.EqualNumpyImplementation(operator)
    #     equal_event = impl.call(event=input_event_data)["event"]

    #     expected_event = NumpyEvent.from_dataframe(new_df)

    #     self.assertEqual(equal_event, expected_event)

    # def test_int_equal(self) -> None:
    #     """Test equal when value is an int."""

    #     df = pd.DataFrame(
    #         [
    #             [1.0, 10],
    #             [2.0, 11],
    #             [3.0, 100],
    #             [4.0, 20],
    #             [5.0, 0],
    #             [6.0, 40],
    #         ],
    #         columns=["timestamp", "weather"],
    #     )

    #     input_event_data = NumpyEvent.from_dataframe(df)

    #     input_event = input_event_data.schema()

    #     new_df = pd.DataFrame(
    #         [
    #             [1.0, True],
    #             [2.0, False],
    #             [3.0, False],
    #             [4.0, False],
    #             [5.0, False],
    #             [6.0, False],
    #         ],
    #         columns=[
    #             "timestamp",
    #             "weather_equal_10",
    #         ],
    #     )

    #     operator = EqualScalarOperator(event=input_event, value=10)
    #     impl = equal.EqualNumpyImplementation(operator)
    #     equal_event = impl.call(event=input_event_data)["event"]

    #     expected_event = NumpyEvent.from_dataframe(new_df)

    #     self.assertEqual(equal_event, expected_event)

    # def test_int_equal_with_int32_column(self) -> None:
    #     """Test equal when value is an int64 and a column is an int32."""

    #     df = pd.DataFrame(
    #         [
    #             [1.0, 10],
    #             [2.0, 11],
    #             [3.0, 100],
    #             [4.0, 20],
    #             [5.0, 0],
    #             [6.0, 40],
    #         ],
    #         columns=["timestamp", "weather"],
    #     )

    #     df["weather"] = df["weather"].astype(np.int32)

    #     input_event_data = NumpyEvent.from_dataframe(df)

    #     input_event = input_event_data.schema()

    #     new_df = pd.DataFrame(
    #         [
    #             [1.0, True],
    #             [2.0, False],
    #             [3.0, False],
    #             [4.0, False],
    #             [5.0, False],
    #             [6.0, False],
    #         ],
    #         columns=[
    #             "timestamp",
    #             "weather_equal_10",
    #         ],
    #     )

    #     # The column is int32, but value is int64. It should still work.
    #     operator = EqualScalarOperator(event=input_event, value=10)
    #     impl = equal.EqualNumpyImplementation(operator)
    #     equal_event = impl.call(event=input_event_data)["event"]

    #     expected_event = NumpyEvent.from_dataframe(new_df)

    #     self.assertEqual(equal_event, expected_event)

    # def test_float_equal_with_float32_column(self) -> None:
    #     """Test equal when value is an int64 and a column is an int32."""

    #     df = pd.DataFrame(
    #         [
    #             [1.0, 10.0],
    #             [2.0, np.nan],
    #             [3.0, 12.0],
    #             [4.0, 13.0],
    #             [5.0, 14.0],
    #             [6.0, 15.0],
    #         ],
    #         columns=["timestamp", "sales"],
    #     )

    #     df["sales"] = df["sales"].astype(np.float32)

    #     input_event_data = NumpyEvent.from_dataframe(df)

    #     input_event = input_event_data.schema()

    #     new_df = pd.DataFrame(
    #         [
    #             [1.0, True],
    #             [2.0, False],
    #             [3.0, False],
    #             [4.0, False],
    #             [5.0, False],
    #             [6.0, False],
    #         ],
    #         columns=[
    #             "timestamp",
    #             "sales_equal_10.0",
    #         ],
    #     )

    #     # The column is float32, but value is float64. It should still work.
    #     operator = EqualScalarOperator(event=input_event, value=10.0)
    #     impl = equal.EqualNumpyImplementation(operator)
    #     equal_event = impl.call(event=input_event_data)["event"]

    #     expected_event = NumpyEvent.from_dataframe(new_df)

    #     self.assertEqual(equal_event, expected_event)

    # def test_value_unsupported_type(self):
    #     """Test operator raises error when value is not a supported type."""

    #     class MyClass:
    #         def __init__(self, a_variable) -> None:
    #             self.a_variable = a_variable

    #     value = MyClass(10)

    #     with self.assertRaises(ValueError):
    #         operator = EqualScalarOperator(event=self.input_event, value=value)


if __name__ == "__main__":
    absltest.main()
