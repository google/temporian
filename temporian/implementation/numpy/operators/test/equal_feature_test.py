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
        """Test equal when event_1 and event_2 are float64."""
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

    def test_string_equal(self) -> None:
        """Test equal when event_1 and event_2 are strings."""
        df_1 = pd.DataFrame(
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

        input_event_1_data = NumpyEvent.from_dataframe(df_1)

        input_event_1 = input_event_1_data.schema()

        df_2 = pd.DataFrame(
            [
                [1.0, "A"],
                [2.0, "X"],
                [3.0, "X"],
                [4.0, "B"],
                [5.0, "10.00"],
                [6.0, "100"],
            ],
            columns=["timestamp", "old_costs"],
        )

        input_event_2_data = NumpyEvent.from_dataframe(df_2)

        input_event_2 = input_event_2_data.schema()
        input_event_2.sampling = input_event_1.sampling

        new_df = pd.DataFrame(
            [
                [1.0, True],
                [2.0, False],
                [3.0, False],
                [4.0, True],
                [5.0, False],
                [6.0, False],
            ],
            columns=[
                "timestamp",
                "costs_equal_old_costs",
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

    def test_int_equal(self) -> None:
        """Test equal when event_1 and event_2 are int64."""

        df_1 = pd.DataFrame(
            [
                [1.0, 1],
                [2.0, 2],
                [3.0, 3],
                [4.0, 4],
                [5.0, 5],
                [6.0, 6],
            ],
            columns=["timestamp", "column_1"],
        )

        input_event_1_data = NumpyEvent.from_dataframe(df_1)

        input_event_1 = input_event_1_data.schema()

        df_2 = pd.DataFrame(
            [
                [1.0, -1],
                [2.0, -2],
                [3.0, -3],
                [4.0, 4],
                [5.0, 5],
                [6.0, 6],
            ],
            columns=["timestamp", "column_2"],
        )

        input_event_2_data = NumpyEvent.from_dataframe(df_2)

        input_event_2 = input_event_2_data.schema()
        input_event_2.sampling = input_event_1.sampling

        new_df = pd.DataFrame(
            [
                [1.0, False],
                [2.0, False],
                [3.0, False],
                [4.0, True],
                [5.0, True],
                [6.0, True],
            ],
            columns=[
                "timestamp",
                "column_1_equal_column_2",
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

    def test_int_equal_with_int32_column(self) -> None:
        """Test equal when event_1 has an int64 feature and event_2 has an int32
        feature.
        """

        df_1 = pd.DataFrame(
            [
                [1.0, 1],
                [2.0, 2],
                [3.0, 3],
                [4.0, 4],
                [5.0, 5],
                [6.0, 6],
            ],
            columns=["timestamp", "column_1"],
        )

        input_event_1_data = NumpyEvent.from_dataframe(df_1)

        input_event_1 = input_event_1_data.schema()

        df_2 = pd.DataFrame(
            [
                [1.0, -1],
                [2.0, -2],
                [3.0, -3],
                [4.0, 4],
                [5.0, 5],
                [6.0, 6],
            ],
            columns=["timestamp", "column_2"],
        )

        df_2["column_2"] = df_2["column_2"].astype("int32")

        input_event_2_data = NumpyEvent.from_dataframe(df_2)

        input_event_2 = input_event_2_data.schema()
        input_event_2.sampling = input_event_1.sampling

        new_df = pd.DataFrame(
            [
                [1.0, False],
                [2.0, False],
                [3.0, False],
                [4.0, True],
                [5.0, True],
                [6.0, True],
            ],
            columns=[
                "timestamp",
                "column_1_equal_column_2",
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

    def test_float_equal_with_float32_column(self) -> None:
        """Test equal when event_1 has a float64 feature and event_2 has a
        float32 feature.
        """

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

        df_2["costs"] = df_2["costs"].astype("float32")

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

    def test_equal_with_different_dtypes_in_event_1(self):
        """Test operator raises error when event_1 has multiple dtypes."""

        df_1 = pd.DataFrame(
            [
                [1.0, 10.0, 10],
                [2.0, np.nan, 0],
                [3.0, 12.0, 12],
                [4.0, 13.0, 13],
                [5.0, 14.0, 14],
                [6.0, 15.0, 15],
            ],
            columns=["timestamp", "sales", "sales_int"],
        )

        input_event_1_data = NumpyEvent.from_dataframe(df_1)

        input_event_1 = input_event_1_data.schema()

        df_2 = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
                [3.0, 12.0],
                [4.0, 13.0],
                [5.0, 14.0],
                [6.0, 15.0],
            ],
            columns=["timestamp", "costs"],
        )

        input_event_2_data = NumpyEvent.from_dataframe(df_2)

        input_event_2 = input_event_2_data.schema()
        input_event_2.sampling = input_event_1.sampling

        with self.assertRaises(ValueError):
            operator = EqualFeatureOperator(
                event_1=input_event_1, event_2=input_event_2
            )

    def test_event_2_with_multiple_features(self):
        """Test operator raises error when event_2 has multiple features."""

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
                [1.0, 10.0, 1.0],
                [2.0, np.nan, 3.2],
                [3.0, 12.0, 4.0],
                [4.0, 13.0, 5.0],
                [5.0, 14.0, 6.0],
                [6.0, 15.0, 7.0],
            ],
            columns=["timestamp", "costs", "costs_2"],
        )

        input_event_2_data = NumpyEvent.from_dataframe(df_2)

        input_event_2 = input_event_2_data.schema()
        input_event_2.sampling = input_event_1.sampling

        with self.assertRaises(ValueError):
            operator = EqualFeatureOperator(
                event_1=input_event_1, event_2=input_event_2
            )

    def test_event_1_and_event_2_different_dtypes(self):
        """Test operator raises error when event_1 and event_2 have different
        dtypes.
        """

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
                [1.0, 10],
                [2.0, 0],
                [3.0, 12],
                [4.0, 13],
                [5.0, 14],
                [6.0, 15],
            ],
            columns=["timestamp", "costs"],
        )

        input_event_2_data = NumpyEvent.from_dataframe(df_2)

        input_event_2 = input_event_2_data.schema()
        input_event_2.sampling = input_event_1.sampling

        with self.assertRaises(ValueError):
            operator = EqualFeatureOperator(
                event_1=input_event_1, event_2=input_event_2
            )


if __name__ == "__main__":
    absltest.main()
