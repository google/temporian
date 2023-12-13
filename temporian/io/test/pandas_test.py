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

import datetime
import math

import numpy as np
import pandas as pd
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set
from temporian.io.pandas import from_pandas, to_pandas
from temporian.test.utils import assertEqualDFRandomRowOrder


class DataFrameToEventTest(absltest.TestCase):
    def test_correct(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            indexes=["product_id"],
        )

        evset = from_pandas(df, indexes=["product_id"], timestamps="timestamp")

        self.assertEqual(evset, expected_evset)

    def test_timestamp_order(self) -> None:
        df = pd.DataFrame(
            [
                [1.0, 100.0],
                [3.0, 300.0],
                [2.0, 200.0],
            ],
            columns=["timestamp", "costs"],
        )

        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={"costs": [100.0, 200.0, 300.0]},
        )
        evset = from_pandas(df)

        self.assertEqual(evset, expected_evset)

    def test_string_column(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, "740"],
                [666964, 2.0, np.nan],
                [574016, 3.0, ""],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": ["740", "nan", ""],
            },
            indexes=["product_id"],
        )

        evset = from_pandas(df, indexes=["product_id"], timestamps="timestamp")

        self.assertEqual(evset, expected_evset)

    def test_multiple_string_formats(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, "740", "A", "D"],
                [666964, 2.0, "400", "B", "E"],
                [574016, 3.0, "200", "C", "F"],
            ],
            columns=["product_id", "timestamp", "str_0", "str_1", "str_2"],
        )
        # set dtype of column costs to string
        df["str_0"] = df["str_0"].astype(str)

        # set dtype of column sales to pandas string
        df["str_1"] = df["str_1"].astype("string")

        # set dtype of column sales2 to np.string_
        df["str_2"] = df["str_2"].astype(np.string_)

        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "str_0": ["740", "400", "200"],
                "str_1": ["A", "B", "C"],
                "str_2": ["D", "E", "F"],
            },
            indexes=["product_id"],
        )
        evset = from_pandas(df, indexes=["product_id"], timestamps="timestamp")

        self.assertEqual(evset, expected_evset)

    def test_string_in_index(self):
        evset = from_pandas(
            pd.DataFrame(
                [
                    ["X1", "Y1", 1.0, 10.0],
                    ["X1", "Y1", 2.0, 11.0],
                    ["X1", "Y1", 3.0, 12.0],
                    ["X2", "Y1", 1.1, 13.0],
                    ["X2", "Y1", 2.1, 14.0],
                    ["X2", "Y1", 3.1, 15.0],
                    ["X2", "Y2", 1.2, 16.0],
                    ["X2", "Y2", 2.2, 17.0],
                    ["X2", "Y2", 3.2, 18.0],
                ],
                columns=["index_x", "index_y", "timestamp", "costs"],
            ),
            indexes=["index_x", "index_y"],
        )
        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            features={
                "costs": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                "index_x": [
                    "X1",
                    "X1",
                    "X1",
                    "X2",
                    "X2",
                    "X2",
                    "X2",
                    "X2",
                    "X2",
                ],
                "index_y": [
                    "Y1",
                    "Y1",
                    "Y1",
                    "Y1",
                    "Y1",
                    "Y1",
                    "Y2",
                    "Y2",
                    "Y2",
                ],
            },
            indexes=["index_x", "index_y"],
        )
        self.assertEqual(evset, expected_evset)

    def test_missing_values(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, np.nan, 573.0],
            },
            indexes=["product_id"],
        )

        evset = from_pandas(df, indexes=["product_id"], timestamps="timestamp")
        self.assertEqual(evset, expected_evset)
        self.assertFalse(evset.schema.is_unix_timestamp)

    def test_npdatetime64_index(self) -> None:
        df = pd.DataFrame(
            [
                [666964, np.datetime64("2022-01-01"), 740.0],
                [666964, np.datetime64("2022-01-02"), 508.0],
                [574016, np.datetime64("2022-01-03"), 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = event_set(
            timestamps=[1640995200, 1641081600, 1641168000],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            indexes=["product_id"],
            is_unix_timestamp=True,
        )
        evset = from_pandas(df, indexes=["product_id"], timestamps="timestamp")
        # validate
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.schema.is_unix_timestamp)

    def test_pdTimestamp_index(self) -> None:
        df = pd.DataFrame(
            [
                [666964, pd.Timestamp("2022-01-01"), 740.0],
                [666964, pd.Timestamp("2022-01-02"), 508.0],
                [574016, pd.Timestamp("2022-01-03"), 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = event_set(
            timestamps=[1640995200, 1641081600, 1641168000],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            indexes=["product_id"],
            is_unix_timestamp=True,
        )
        evset = from_pandas(df, indexes=["product_id"], timestamps="timestamp")
        # validate
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.schema.is_unix_timestamp)

    def test_string_timestamp(self) -> None:
        df = pd.DataFrame(
            [
                [666964, "2022-01-01", 740.0],
                [666964, "2022-01-02", 508.0],
                [574016, "2022-01-03", 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        expected_evset = event_set(
            timestamps=[1640995200, 1641081600, 1641168000],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            indexes=["product_id"],
            is_unix_timestamp=True,
        )

        evset = from_pandas(df, indexes=["product_id"], timestamps="timestamp")

        # validate
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.schema.is_unix_timestamp)

    def test_datetime_index(self) -> None:
        df = pd.DataFrame(
            [
                [
                    666964,
                    datetime.datetime.strptime("2022-01-01", "%Y-%m-%d"),
                    740.0,
                ],
                [
                    666964,
                    datetime.datetime.strptime("2022-01-02", "%Y-%m-%d"),
                    508.0,
                ],
                [
                    574016,
                    datetime.datetime.strptime("2022-01-03", "%Y-%m-%d"),
                    573.0,
                ],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = event_set(
            timestamps=[1640995200, 1641081600, 1641168000],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            indexes=["product_id"],
            is_unix_timestamp=True,
        )

        evset = from_pandas(df, indexes=["product_id"], timestamps="timestamp")
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.schema.is_unix_timestamp)

    def test_invalid_boolean_timestamp_type(self) -> None:
        df = pd.DataFrame(
            [
                [666964, False, 740.0],
                [666964, True, 508.0],
                [574016, False, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        with self.assertRaises(ValueError):
            from_pandas(df, indexes=["product_id"], timestamps="timestamp")

    def test_invalid_string_timestamp_type(self) -> None:
        df = pd.DataFrame(
            [
                [666964, "A", 740.0],
                [666964, "B", 508.0],
                [574016, 1.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        with self.assertRaises(ValueError):
            from_pandas(df, indexes=["product_id"], timestamps="timestamp")

    def test_timestamps_with_missing_values(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
                [574016, None, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        with self.assertRaises(ValueError):
            from_pandas(df, indexes=["product_id"], timestamps="timestamp")

    def test_timestamps_with_non_supported_object(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
                [574016, object(), 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        with self.assertRaises(ValueError):
            from_pandas(df, indexes=["product_id"], timestamps="timestamp")

    def test_no_index(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
        )
        evset = from_pandas(df, indexes=[], timestamps="timestamp")
        self.assertEqual(evset, expected_evset)

    def test_datetime_in_feature_column(self) -> None:
        df = pd.DataFrame(
            [
                [666964, np.datetime64("2022-01-01"), 740.0],
                [666964, np.datetime64("2022-01-02"), 508.0],
                [574016, np.datetime64("2022-01-03"), 573.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )

        expected_evset = event_set(
            timestamps=[740.0, 508.0, 573.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": (
                    np.array(
                        ["2022-01-01", "2022-01-02", "2022-01-03"],
                        dtype="datetime64[ns]",
                    ).astype(np.float64)
                    / 1e9
                ),
            },
        )
        evset = from_pandas(df, indexes=[], timestamps="timestamp")

        self.assertEqual(evset, expected_evset)

    def test_nan_in_string(self) -> None:
        evset = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3],
                    "x": ["a", math.nan, "b"],
                    "y": [1, 2, 3],
                }
            )
        )

        expected_evset = event_set(
            timestamps=[1, 2, 3],
            features={
                "x": ["a", "nan", "b"],
                "y": [1, 2, 3],
            },
        )

        self.assertEqual(evset, expected_evset)

    def test_evset_to_df(self) -> None:
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            indexes=["product_id"],
        )

        expected_df = pd.DataFrame(
            [
                [666964, 740.0, 1.0],
                [574016, 573.0, 3.0],
                [666964, 508.0, 2.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )
        df = to_pandas(evset)

        assertEqualDFRandomRowOrder(self, df, expected_df)

    def test_evset_to_df_no_index(self) -> None:
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
        )

        expected_df = pd.DataFrame(
            [
                [666964, 740.0, 1.0],
                [666964, 508.0, 2.0],
                [574016, 573.0, 3.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )
        df = to_pandas(evset)

        assertEqualDFRandomRowOrder(self, df, expected_df)

    def test_evset_to_df_multiple_index(self) -> None:
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            features={
                "sma_a": [10.0, 10.5, 11.0, 13.0, 13.5, 14.0, 16.0, 16.5, 17.0],
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
            },
            indexes=["x", "y"],
        )
        expected_df = pd.DataFrame(
            [
                ["X1", "Y1", 10.0, 1.0],
                ["X1", "Y1", 10.5, 2.0],
                ["X1", "Y1", 11.0, 3.0],
                ["X2", "Y1", 13.0, 1.1],
                ["X2", "Y1", 13.5, 2.1],
                ["X2", "Y1", 14.0, 3.1],
                ["X2", "Y2", 16.0, 1.2],
                ["X2", "Y2", 16.5, 2.2],
                ["X2", "Y2", 17.0, 3.2],
            ],
            columns=["x", "y", "sma_a", "timestamp"],
        )
        df = to_pandas(evset)

        assertEqualDFRandomRowOrder(self, df, expected_df)

    def test_evset_to_df_string_feature(self) -> None:
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": ["740.0", "508.0", "573.0"],
            },
            indexes=["product_id"],
        )
        expected_df = pd.DataFrame(
            [
                [666964, "740.0", 1.0],
                [666964, "508.0", 2.0],
                [574016, "573.0", 3.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )
        df = to_pandas(evset)

        assertEqualDFRandomRowOrder(self, df, expected_df)

    def test_evset_to_df_unix_timestamp(self):
        evset = event_set(
            timestamps=[
                datetime.datetime(2023, 11, 1),
                datetime.datetime(2023, 11, 2),
                datetime.datetime(2023, 11, 3),
            ],
            features={
                "f": [1, 2, 3],
            },
        )

        df = to_pandas(evset)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

        evset2 = from_pandas(df)
        assert evset2.schema.is_unix_timestamp

    def test_timestamps_params(self):
        evset = event_set(
            timestamps=[
                datetime.datetime(2023, 11, 1),
                datetime.datetime(2023, 11, 2),
                datetime.datetime(2023, 11, 3),
            ],
            features={
                "f": [1, 2, 3],
            },
        )
        df = to_pandas(evset)
        assert "timestamp" in df.columns
        assert np.issubdtype(df["timestamp"].dtype, np.datetime64)

        df = to_pandas(evset, timestamp_to_datetime=False)
        assert "timestamp" in df.columns
        assert np.issubdtype(df["timestamp"].dtype, np.float64)

        df = to_pandas(evset, timestamps=False)
        assert "timestamp" not in df.columns


if __name__ == "__main__":
    absltest.main()
