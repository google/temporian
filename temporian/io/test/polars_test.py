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
import polars as pl
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set
from temporian.io.polars import to_polars, from_polars
from temporian.test.utils import assertEqualDFRandomRowOrderPolars


class DataFrameToEventTest(absltest.TestCase):
    def test_correct(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": [1.0, 2.0, 3.0],
                "costs": [740.0, 508.0, 573.0],
            }
        )
        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            indexes=["product_id"],
        )
        evset = from_polars(df, indexes=["product_id"], timestamps="timestamp")
        self.assertEqual(evset, expected_evset)

    def test_timestamp_order(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "costs": [100.0, 200.0, 300.0],
            }
        )
        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={"costs": [100.0, 200.0, 300.0]},
        )
        evset = from_polars(df)

        self.assertEqual(evset, expected_evset)

    def test_string_column_with_polars(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": [1.0, 2.0, 3.0],
                "costs": ["740", None, ""],
            }
        )

        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": ["740", None, ""],
            },
            indexes=["product_id"],
        )

        evset = from_polars(df, indexes=["product_id"], timestamps="timestamp")

        self.assertEqual(evset, expected_evset)

    def test_string_in_index(self):
        evset = from_polars(
            pl.DataFrame(
                {
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
                    "timestamp": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
                    "costs": [
                        10.0,
                        11.0,
                        12.0,
                        13.0,
                        14.0,
                        15.0,
                        16.0,
                        17.0,
                        18.0,
                    ],
                },
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

    def test_missing_values_with_polars(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": [1.0, 2.0, 3.0],
                "costs": [740.0, None, 573.0],
            }
        )

        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, np.nan, 573.0],
            },
            indexes=["product_id"],
        )

        evset = from_polars(df, indexes=["product_id"], timestamps="timestamp")
        self.assertEqual(evset, expected_evset)

    def test_npdatetime64_index(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": [
                    np.datetime64("2022-01-01"),
                    np.datetime64("2022-01-02"),
                    np.datetime64("2022-01-03"),
                ],
                "timestamp": ["2022-01-01", "2022-01-02", "2022-01-03"],
                "costs": [740.0, 508.0, 573.0],
            }
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
        evset = from_polars(df, indexes=["product_id"], timestamps="timestamp")

        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.schema.is_unix_timestamp)

    def test_plTimestamp_index(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": ["2022-01-01", "2022-01-02", "2022-01-03"],
                "costs": [740.0, 508.0, 573.0],
            }
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
        evset = from_polars(df, indexes=["product_id"], timestamps="timestamp")
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.schema.is_unix_timestamp)

    def test_string_timestamp(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": ["2022-01-01", "2022-01-02", "2022-01-03"],
                "costs": [740.0, 508.0, 573.0],
            }
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

        evset = from_polars(df, indexes=["product_id"], timestamps="timestamp")

        # validate
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.schema.is_unix_timestamp)

    def test_datetime_index(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": [
                    datetime.datetime.strptime("2022-01-01", "%Y-%m-%d"),
                    datetime.datetime.strptime("2022-01-02", "%Y-%m-%d"),
                    datetime.datetime.strptime("2022-01-03", "%Y-%m-%d"),
                ],
                "costs": [740.0, 508.0, 573.0],
            }
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

        evset = from_polars(df, indexes=["product_id"], timestamps="timestamp")
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.schema.is_unix_timestamp)

    def test_invalid_boolean_timestamp_type(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": [True, False, True],
                "costs": [740.0, 508.0, 573.0],
            }
        )

        with self.assertRaises(ValueError):
            from_polars(df, indexes=["product_id"], timestamps="timestamp")

    def test_invalid_string_timestamp_type(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": ["A", "B", 1.0],
                "costs": [740.0, 508.0, 573.0],
            }
        )

        with self.assertRaises(ValueError):
            from_polars(df, indexes=["product_id"], timestamps="timestamp")

    def test_timestamps_with_missing_values(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016, 574016],
                "timestamp": [1.0, 2.0, 3.0, None],
                "costs": [740.0, 508.0, 573.0, 573.0],
            }
        )

        with self.assertRaises(ValueError):
            from_polars(df, indexes=["product_id"], timestamps="timestamp")

    def test_timestamps_with_non_supported_object(self) -> None:
        with self.assertRaises(TypeError):
            df = pl.DataFrame(
                {
                    "product_id": [666964, 666964, 574016, 574016],
                    "timestamp": [1.0, 2.0, 3.0, object()],
                    "costs": [740.0, 508.0, 573.0, 573.0],
                }
            )

    def test_no_index(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "timestamp": [1.0, 2.0, 3.0],
                "costs": [740.0, 508.0, 573.0],
            }
        )
        expected_evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
        )
        evset = from_polars(df, indexes=[], timestamps="timestamp")
        self.assertEqual(evset, expected_evset)

    def test_datetime_in_feature_column(self) -> None:
        df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "costs": ["2022-01-01", "2022-01-02", "2022-01-03"],
                "timestamp": [740.0, 508.0, 573.0],
            }
        )
        expected_evset = event_set(
            timestamps=[740.0, 508.0, 573.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": ["2022-01-01", "2022-01-02", "2022-01-03"],
            },
        )
        evset = from_polars(df, indexes=[], timestamps="timestamp")
        self.assertEqual(evset, expected_evset)

    def test_nan_in_string(self) -> None:
        evset = from_polars(
            pl.DataFrame(
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
                "x": ["a", "None", "b"],
                "y": [1, 2, 3],
            },
        )

        self.assertEqual(evset, expected_evset)

    def test_evset_to_df(self):
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            indexes=["product_id"],
        )

        expected_df = pl.DataFrame(
            {
                "product_id": [666964, 574016, 666964],
                "costs": [740.0, 573.0, 508.0],
                "timestamp": [1.0, 3.0, 2.0],
            }
        )

        df = to_polars(evset)
        assertEqualDFRandomRowOrderPolars(self, df, expected_df)

    def test_evset_to_df_no_index(self):
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
        )

        expected_df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
                "timestamp": [1.0, 2.0, 3.0],
            }
        )

        df = to_polars(evset)
        assertEqualDFRandomRowOrderPolars(self, df, expected_df)

    def test_evset_to_df_multiple_index(self):
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            features={
                "sma_a": [10.0, 10.5, 11.0, 13.0, 13.5, 14.0, 16.0, 16.5, 17.0],
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
            },
            indexes=["x", "y"],
        )

        expected_df = pl.DataFrame(
            {
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
                "sma_a": [10.0, 10.5, 11.0, 13.0, 13.5, 14.0, 16.0, 16.5, 17.0],
                "timestamp": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            }
        )

        df = to_polars(evset)
        assertEqualDFRandomRowOrderPolars(self, df, expected_df)

    def test_evset_to_df_string_feature(self):
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": ["740.0", "508.0", "573.0"],
            },
            indexes=["product_id"],
        )

        expected_df = pl.DataFrame(
            {
                "product_id": [666964, 666964, 574016],
                "costs": ["740.0", "508.0", "573.0"],
                "timestamp": [1.0, 2.0, 3.0],
            }
        )

        df = to_polars(evset)
        assertEqualDFRandomRowOrderPolars(self, df, expected_df)

    def test_evset_to_df_unix_timestamp(self):
        evset = event_set(
            timestamps=[
                datetime.datetime(2023, 11, 1),
                datetime.datetime(2023, 11, 2),
                datetime.datetime(2023, 11, 3),
            ],
            features={"f": [1, 2, 3]},
        )

        # Assuming to_polars correctly handles datetime conversion
        df = to_polars(evset)
        self.assertTrue(df["timestamp"].dtype == pl.Datetime)
        evset2 = from_polars(df)
        assert evset2.schema.is_unix_timestamp

    def test_timestamps_params(self):
        evset = event_set(
            timestamps=[
                datetime.datetime(2023, 11, 1),
                datetime.datetime(2023, 11, 2),
                datetime.datetime(2023, 11, 3),
            ],
            features={"f": [1, 2, 3]},
        )

        df = to_polars(evset)
        self.assertTrue("timestamp" in df.columns)
        self.assertTrue(df["timestamp"].dtype == pl.Datetime)


if __name__ == "__main__":
    absltest.main()
