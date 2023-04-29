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

import pandas as pd
import numpy as np
from absl.testing import absltest
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent


class EventToDataFrameTest(absltest.TestCase):
    def test_numpy_event_to_df(self) -> None:
        numpy_event = NumpyEvent(
            data={
                (666964,): IndexData(
                    features=[np.array([740.0, 508.0])],
                    timestamps=np.array([1.0, 2.0]),
                ),
                (574016,): IndexData(
                    features=[np.array([573.0])], timestamps=np.array([3.0])
                ),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=False,
        )
        expected_df = pd.DataFrame(
            [
                [666964, 740.0, 1.0],
                [666964, 508.0, 2.0],
                [574016, 573.0, 3.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )
        df = numpy_event.to_dataframe()

        # validate
        self.assertTrue(df.equals(expected_df))

    def test_numpy_event_to_df_with_datetimes(self) -> None:
        numpy_event = NumpyEvent(
            data={
                (666964,): IndexData(
                    features=[np.array([740.0, 508.0])],
                    timestamps=np.array([1, 2]),
                ),
                (574016,): IndexData(
                    features=[np.array([573.0])], timestamps=np.array([3])
                ),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=False,
        )
        expected_df = pd.DataFrame(
            [
                # use timestamp as datetime from numpy
                [666964, 740.0, 1.0],
                [666964, 508.0, 2.0],
                [574016, 573.0, 3.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )
        df = numpy_event.to_dataframe()

        # validate
        self.assertTrue(df.equals(expected_df))

    def test_numpy_event_to_df_no_index(self) -> None:
        numpy_event = NumpyEvent(
            data={
                (): IndexData(
                    features=[
                        np.array([666964, 666964, 574016]),
                        np.array([740.0, 508.0, 573.0]),
                    ],
                    timestamps=np.array([1.0, 2.0, 3.0]),
                ),
            },
            feature_names=["product_id", "costs"],
            index_names=[],
            is_unix_timestamp=False,
        )
        expected_df = pd.DataFrame(
            [
                [666964, 740.0, 1.0],
                [666964, 508.0, 2.0],
                [574016, 573.0, 3.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )
        df = numpy_event.to_dataframe()

        # validate
        self.assertTrue(df.equals(expected_df))

    def test_numpy_event_to_df_multiple_index(self) -> None:
        numpy_event = NumpyEvent(
            data={
                ("X1", "Y1"): IndexData(
                    features=[np.array([10.0, 10.5, 11.0])],
                    timestamps=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                ),
                ("X2", "Y1"): IndexData(
                    features=[np.array([13.0, 13.5, 14.0])],
                    timestamps=np.array([1.1, 2.1, 3.1], dtype=np.float64),
                ),
                ("X2", "Y2"): IndexData(
                    features=[np.array([16.0, 16.5, 17.0])],
                    timestamps=np.array([1.2, 2.2, 3.2], dtype=np.float64),
                ),
            },
            feature_names=["sma_a"],
            index_names=["x", "y"],
            is_unix_timestamp=False,
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
        df = numpy_event.to_dataframe()

        # validate
        self.assertTrue(df.equals(expected_df))

    def test_numpy_event_to_df_string_feature(self) -> None:
        numpy_event = NumpyEvent(
            data={
                (666964,): IndexData(
                    features=[np.array(["740.0", "508.0"]).astype(np.str_)],
                    timestamps=np.array([1.0, 2.0]),
                ),
                (574016,): IndexData(
                    features=[np.array(["573.0"]).astype(np.str_)],
                    timestamps=np.array([3.0]),
                ),
            },
            feature_names=["costs"],
            index_names=["product_id"],
            is_unix_timestamp=False,
        )
        expected_df = pd.DataFrame(
            [
                [666964, "740.0", 1.0],
                [666964, "508.0", 2.0],
                [574016, "573.0", 3.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )
        df = numpy_event.to_dataframe()

        # validate
        self.assertTrue(df.equals(expected_df))


if __name__ == "__main__":
    absltest.main()
