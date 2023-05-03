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

from absl.testing import absltest
import numpy as np
import pandas as pd

from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.event_set import IndexData


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
        expected_evset = EventSet(
            data={
                (666964,): IndexData(
                    [np.array([740.0, 508.0])], np.array([1.0, 2.0])
                ),
                (574016,): IndexData([np.array([573.0])], np.array([3.0])),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=False,
        )
        evset = EventSet.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )
        # validate
        self.assertTrue(evset == expected_evset)

    def test_timestamp_order(self) -> None:
        df = pd.DataFrame(
            [
                [1.0, 100.0],
                [3.0, 300.0],
                [2.0, 200.0],
            ],
            columns=["timestamp", "costs"],
        )
        expected_evset = EventSet(
            data={
                (): IndexData(
                    [np.array([100.0, 200.0, 300.0])], np.array([1.0, 2.0, 3.0])
                ),
            },
            feature_names=["costs"],
            index_names=[],
            is_unix_timestamp=False,
        )
        evset = EventSet.from_dataframe(df)

        # validate
        self.assertTrue(evset == expected_evset)

    def test_string_column(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, "740"],
                [666964, 2.0, "B"],
                [574016, 3.0, ""],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = EventSet(
            data={
                (666964,): IndexData(
                    [np.array(["740", "B"])], np.array([1.0, 2.0])
                ),
                (574016,): IndexData([np.array([""])], np.array([3.0])),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=False,
        )
        evset = EventSet.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )
        # validate
        self.assertTrue(evset == expected_evset)

    def test_mixed_types_in_string_column(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, "740", "A"],
                [666964, 2.0, "400", 101],
                [574016, 3.0, np.nan, "B"],
            ],
            columns=["product_id", "timestamp", "costs", "sales"],
        )
        # not allowed
        with self.assertRaises(ValueError):
            EventSet.from_dataframe(
                df, index_names=["product_id"], timestamp_column="timestamp"
            )

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

        expected_evset = EventSet(
            data={
                (666964,): IndexData(
                    [
                        np.array(["740", "400"]),
                        np.array(["A", "B"]),
                        np.array(["D", "E"]),
                    ],
                    np.array([1.0, 2.0]),
                ),
                (574016,): IndexData(
                    [
                        np.array(["200"]),
                        np.array(["C"]),
                        np.array(["F"]),
                    ],
                    np.array([3.0]),
                ),
            },
            feature_names=["str_0", "str_1", "str_2"],
            index_names=["product_id"],
            is_unix_timestamp=False,
        )
        evset = EventSet.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )
        # validate
        self.assertTrue(evset == expected_evset)

    def test_string_in_index(self):
        evset = EventSet.from_dataframe(
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
            index_names=["index_x", "index_y"],
        )
        expected_evset = EventSet(
            data={
                ("X1", "Y1"): IndexData(
                    [np.array([10.0, 11.0, 12.0])], np.array([1.0, 2.0, 3.0])
                ),
                ("X2", "Y1"): IndexData(
                    [np.array([13.0, 14.0, 15.0])], np.array([1.1, 2.1, 3.1])
                ),
                ("X2", "Y2"): IndexData(
                    [np.array([16.0, 17.0, 18.0])], np.array([1.2, 2.2, 3.2])
                ),
            },
            feature_names=["costs"],
            index_names=["index_x", "index_y"],
            is_unix_timestamp=False,
        )
        # validate
        self.assertTrue(evset == expected_evset)

    def test_missing_values(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = EventSet(
            data={
                (666964,): IndexData(
                    [np.array([740.0, np.nan])], np.array([1.0, 2.0])
                ),
                (574016,): IndexData([np.array([573.0])], np.array([3.0])),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=False,
        )
        evset = EventSet.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )
        # validate
        self.assertTrue(evset == expected_evset)
        self.assertFalse(evset.is_unix_timestamp)

    def test_npdatetime64_index(self) -> None:
        df = pd.DataFrame(
            [
                [666964, np.datetime64("2022-01-01"), 740.0],
                [666964, np.datetime64("2022-01-02"), 508.0],
                [574016, np.datetime64("2022-01-03"), 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = EventSet(
            data={
                (666964,): IndexData(
                    [np.array([740.0, 508.0])],
                    np.array([1640995200, 1641081600]),
                ),
                (574016,): IndexData(
                    [np.array([573.0])], np.array([1641168000])
                ),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=True,
        )
        evset = EventSet.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )
        # validate
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.is_unix_timestamp)

    def test_pdTimestamp_index(self) -> None:
        df = pd.DataFrame(
            [
                [666964, pd.Timestamp("2022-01-01"), 740.0],
                [666964, pd.Timestamp("2022-01-02"), 508.0],
                [574016, pd.Timestamp("2022-01-03"), 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = EventSet(
            data={
                (666964,): IndexData(
                    [np.array([740.0, 508.0])],
                    np.array([1640995200, 1641081600]),
                ),
                (574016,): IndexData(
                    [np.array([573.0])], np.array([1641168000])
                ),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=True,
        )
        evset = EventSet.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )
        # validate
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.is_unix_timestamp)

    def test_string_timestamp(self) -> None:
        df = pd.DataFrame(
            [
                [666964, "2022-01-01", 740.0],
                [666964, "2022-01-02", 508.0],
                [574016, "2022-01-03", 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        expected_evset = EventSet(
            data={
                (666964,): IndexData(
                    [np.array([740.0, 508.0])],
                    np.array([1640995200, 1641081600]),
                ),
                (574016,): IndexData(
                    [np.array([573.0])], np.array([1641168000])
                ),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=True,
        )

        evset = EventSet.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )

        # validate
        self.assertEqual(evset, expected_evset)
        self.assertTrue(evset.is_unix_timestamp)

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
        expected_evset = EventSet(
            data={
                (666964,): IndexData(
                    [np.array([740.0, 508.0])],
                    np.array([1640995200, 1641081600]),
                ),
                (574016,): IndexData(
                    [np.array([573.0])], np.array([1641168000])
                ),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=True,
        )
        evset = EventSet.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )
        # validate
        self.assertTrue(evset == expected_evset)
        self.assertTrue(evset.is_unix_timestamp)

    def test_invalid_boolean_timestamp_type(self) -> None:
        df = pd.DataFrame(
            [
                [666964, False, 740.0],
                [666964, True, 508.0],
                [574016, False, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        with self.assertRaises(TypeError):
            EventSet.from_dataframe(
                df, index_names=["product_id"], timestamp_column="timestamp"
            )

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
            EventSet.from_dataframe(
                df, index_names=["product_id"], timestamp_column="timestamp"
            )

    def test_timestamp_column_with_missing_values(self) -> None:
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
            EventSet.from_dataframe(
                df, index_names=["product_id"], timestamp_column="timestamp"
            )

    def test_timestamp_column_with_non_supported_object(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
                [574016, object(), 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        with self.assertRaises(TypeError):
            EventSet.from_dataframe(
                df, index_names=["product_id"], timestamp_column="timestamp"
            )

    def test_no_index(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )
        expected_evset = EventSet(
            data={
                (): IndexData(
                    [
                        np.array([666964, 666964, 574016]),
                        np.array([740.0, 508.0, 573.0]),
                    ],
                    np.array([1.0, 2.0, 3.0]),
                ),
            },
            feature_names=["product_id", "costs"],
            index_names=[],
            is_unix_timestamp=False,
        )
        evset = EventSet.from_dataframe(
            df, index_names=[], timestamp_column="timestamp"
        )
        # validate
        self.assertTrue(evset == expected_evset)

    def test_datetime_in_feature_column(self) -> None:
        df = pd.DataFrame(
            [
                [666964, np.datetime64("2022-01-01"), 740.0],
                [666964, np.datetime64("2022-01-02"), 508.0],
                [574016, np.datetime64("2022-01-03"), 573.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )

        # assert it raises regex value error
        with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
            EventSet.from_dataframe(
                df, index_names=["product_id"], timestamp_column="timestamp"
            )


if __name__ == "__main__":
    absltest.main()
