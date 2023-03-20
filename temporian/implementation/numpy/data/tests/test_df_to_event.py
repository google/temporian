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
import time
import pandas as pd
import numpy as np
from absl.testing import absltest
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
            index=["product_id"],
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
                [666964, 1.0, "740.0"],
                [666964, 2.0, "508.0"],
                [574016, 3.0, "573.0"],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        self.assertRaisesRegex(
            ValueError,
            "Unsupported dtype",
            NumpyEvent.from_dataframe,
            df,
            ["product_id"],
        )

    def test_categorical_column(self) -> None:
        df = pd.DataFrame(
            [
                [666964, 1.0, "A"],
                [666964, 2.0, "B"],
                [574016, 3.0, "A"],
            ],
            columns=["product_id", "timestamp", "costs"],
        )

        # convert costs to categorical
        df["costs"] = df["costs"].astype("category")
        cat_dict = dict(zip(df["costs"].cat.categories, df["costs"].cat.codes))

        numpy_sampling = NumpySampling(
            data={
                (666964,): np.array([1.0, 2.0]),
                (574016,): np.array([3.0]),
            },
            index=["product_id"],
        )

        expected_numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        data=np.array([cat_dict["A"], cat_dict["B"]]).astype(
                            np.int32
                        ),
                        name="costs",
                    )
                ],
                (574016,): [
                    NumpyFeature(
                        data=np.array([cat_dict["A"]]).astype(np.int32),
                        name="costs",
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
            index=["product_id"],
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
            index=["product_id"],
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
            index=["product_id"],
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
            index=["product_id"],
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
            index=[],
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
