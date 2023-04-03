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
import datetime
import numpy as np
import pandas as pd

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


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

        numpy_sampling = NumpySampling(
            data={
                (666964,): np.array([1.0, 2.0]),
                (574016,): np.array([3.0]),
            },
            index={"product_id": np.int64},
        )

        expected_numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(data=np.array([740.0, 508.0]), name="costs")
                ],
                (574016,): [NumpyFeature(data=np.array([573.0]), name="costs")],
            },
            sampling=numpy_sampling,
        )

        numpy_event = NumpyEvent.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )
        # validate
        self.assertTrue(numpy_event == expected_numpy_event)

    def test_string_column(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, "740"],
                [666964, 2.0, "B"],
                [574016, 3.0, ""],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        numpy_sampling = NumpySampling(
            data={
                (666964,): np.array([1.0, 2.0]),
                (574016,): np.array([3.0]),
            },
            index={"product_id": np.int64},
        )

        expected_numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        data=np.array(["740", "B"]).astype(np.str_),
                        name="costs",
                    )
                ],
                (574016,): [
                    NumpyFeature(
                        data=np.array([""]).astype(np.str_), name="costs"
                    )
                ],
            },
            sampling=numpy_sampling,
        )

        numpy_event = NumpyEvent.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )

        # validate
        self.assertTrue(numpy_event == expected_numpy_event)

    def test_mixed_types_in_string_column(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, "740", "A"],
                [666964, 2.0, "400", 101],
                [574016, 3.0, np.nan, "B"],
            ],
            columns=["product_id", "timestamp", "costs", "sales"],
        )

        # Not allowed
        with self.assertRaises(ValueError):
            NumpyEvent.from_dataframe(
                df, index_names=["product_id"], timestamp_column="timestamp"
            )

    def test_multiple_string_formats(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, "740", "A", "D"],
                [666964, 2.0, "400", "B", "E"],
                [574016, 3.0, "200", "C", "F"],
            ],
            columns=["product_id", "timestamp", "costs", "sales", "sales2"],
        )

        # set dtype of column costs to string
        df["costs"] = df["costs"].astype(str)
        # set dtype of column sales to pandas string
        df["sales"] = df["sales"].astype("string")
        # set dtype of column sales2 to np.string_
        df["sales2"] = df["sales2"].astype(np.string_)

        numpy_sampling = NumpySampling(
            data={
                (666964,): np.array([1.0, 2.0]),
                (574016,): np.array([3.0]),
            },
            index={"product_id": np.int64},
        )

        expected_numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        data=np.array(["740", "400"]).astype(np.str_),
                        name="costs",
                    ),
                    NumpyFeature(
                        data=np.array(["A", "B"]).astype(np.str_),
                        name="sales",
                    ),
                    NumpyFeature(
                        data=np.array(["D", "E"]).astype(np.str_),
                        name="sales2",
                    ),
                ],
                (574016,): [
                    NumpyFeature(
                        data=np.array(["200"]).astype(np.str_), name="costs"
                    ),
                    NumpyFeature(
                        data=np.array(["C"]).astype(np.str_), name="sales"
                    ),
                    NumpyFeature(
                        data=np.array(["F"]).astype(np.str_), name="sales2"
                    ),
                ],
            },
            sampling=numpy_sampling,
        )

        numpy_event = NumpyEvent.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )

        # validate
        self.assertTrue(numpy_event == expected_numpy_event)

    def test_string_in_index(self):
        numpy_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    ["X1", "Y1", 10.0, 1],
                    ["X1", "Y1", 11.0, 2],
                    ["X1", "Y1", 12.0, 3],
                    ["X2", "Y1", 13.0, 1.1],
                    ["X2", "Y1", 14.0, 2.1],
                    ["X2", "Y1", 15.0, 3.1],
                    ["X2", "Y2", 16.0, 1.2],
                    ["X2", "Y2", 17.0, 2.2],
                    ["X2", "Y2", 18.0, 3.2],
                ],
                columns=["x", "y", "a", "timestamp"],
            ),
            index_names=["x", "y"],
        )

        expected_numpy_event = NumpyEvent(
            data={
                ("X1", "Y1"): [
                    NumpyFeature(
                        name="a",
                        data=np.array([10.0, 11.0, 12.0]),
                    )
                ],
                ("X2", "Y1"): [
                    NumpyFeature(
                        name="a",
                        data=np.array([13.0, 14.0, 15.0]),
                    )
                ],
                ("X2", "Y2"): [
                    NumpyFeature(
                        name="a",
                        data=np.array([16.0, 17.0, 18.0]),
                    )
                ],
            },
            sampling=NumpySampling(
                index={"x": np.str_, "y": np.str_},
                data={
                    ("X1", "Y1"): np.array([1, 2, 3], dtype=np.float64),
                    ("X2", "Y1"): np.array([1.1, 2.1, 3.1], dtype=np.float64),
                    ("X2", "Y2"): np.array([1.2, 2.2, 3.2], dtype=np.float64),
                },
            ),
        )
        # validate
        self.assertTrue(numpy_event == expected_numpy_event)

    def test_missing_values(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        numpy_sampling = NumpySampling(
            data={
                (666964,): np.array([1.0, 2.0]),
                (574016,): np.array([3.0]),
            },
            index={"product_id": np.int64},
        )

        expected_numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(data=np.array([740.0, np.nan]), name="costs")
                ],
                (574016,): [NumpyFeature(data=np.array([573.0]), name="costs")],
            },
            sampling=numpy_sampling,
        )

        numpy_event = NumpyEvent.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )

        # validate
        self.assertTrue(numpy_event == expected_numpy_event)
        self.assertFalse(numpy_event.sampling.is_unix_timestamp)

    def test_npdatetime64_index(self) -> None:
        df = pd.DataFrame(
            [
                [666964, np.datetime64("2022-01-01"), 740.0],
                [666964, np.datetime64("2022-01-02"), 508.0],
                [574016, np.datetime64("2022-01-03"), 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        # dates converted to timestamp UTC epoch
        numpy_sampling = NumpySampling(
            data={
                (666964,): np.array([1640995200, 1641081600]),
                (574016,): np.array([1641168000]),
            },
            index={"product_id": np.int64},
        )

        expected_numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(data=np.array([740.0, 508.0]), name="costs")
                ],
                (574016,): [NumpyFeature(data=np.array([573.0]), name="costs")],
            },
            sampling=numpy_sampling,
        )

        numpy_event = NumpyEvent.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )

        # validate
        self.assertTrue(numpy_event == expected_numpy_event)
        self.assertTrue(numpy_event.sampling.is_unix_timestamp)

    def test_pdTimestamp_index(self) -> None:
        df = pd.DataFrame(
            [
                [666964, pd.Timestamp("2022-01-01"), 740.0],
                [666964, pd.Timestamp("2022-01-02"), 508.0],
                [574016, pd.Timestamp("2022-01-03"), 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        # dates converted to timestamp UTC epoch
        numpy_sampling = NumpySampling(
            data={
                (666964,): np.array([1640995200, 1641081600]),
                (574016,): np.array([1641168000]),
            },
            index={"product_id": np.int64},
        )

        expected_numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(data=np.array([740.0, 508.0]), name="costs")
                ],
                (574016,): [NumpyFeature(data=np.array([573.0]), name="costs")],
            },
            sampling=numpy_sampling,
        )

        numpy_event = NumpyEvent.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )

        # validate
        self.assertTrue(numpy_event == expected_numpy_event)
        self.assertTrue(numpy_event.sampling.is_unix_timestamp)

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

        # dates converted to timestamp UTC epoch
        numpy_sampling = NumpySampling(
            data={
                (666964,): np.array([1640995200, 1641081600]),
                (574016,): np.array([1641168000]),
            },
            index={"product_id": np.int64},
        )

        expected_numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(data=np.array([740.0, 508.0]), name="costs")
                ],
                (574016,): [NumpyFeature(data=np.array([573.0]), name="costs")],
            },
            sampling=numpy_sampling,
        )

        numpy_event = NumpyEvent.from_dataframe(
            df, index_names=["product_id"], timestamp_column="timestamp"
        )

        # validate
        self.assertTrue(numpy_event == expected_numpy_event)
        self.assertTrue(numpy_event.sampling.is_unix_timestamp)

    def test_no_index(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        numpy_sampling = NumpySampling(
            data={
                (): np.array([1.0, 2.0, 3.0]),
            },
            index={},
        )

        expected_numpy_event = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        data=np.array([666964, 666964, 574016]),
                        name="product_id",
                    ),
                    NumpyFeature(
                        data=np.array([740.0, 508.0, 573.0]), name="costs"
                    ),
                ],
            },
            sampling=numpy_sampling,
        )

        numpy_event = NumpyEvent.from_dataframe(
            df, index_names=[], timestamp_column="timestamp"
        )

        # validate
        self.assertTrue(numpy_event == expected_numpy_event)

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
            NumpyEvent.from_dataframe(
                df, index_names=["product_id"], timestamp_column="timestamp"
            )


if __name__ == "__main__":
    absltest.main()
