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
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


class EventToDataFrameTest(absltest.TestCase):
    def test_numpy_event_to_df(self) -> None:
        numpy_sampling = NumpySampling(
            data={
                (666964,): np.array([1.0, 2.0]),
                (574016,): np.array([3.0]),
            },
            index=["product_id"],
        )

        numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(data=np.array([740.0, 508.0]), name="costs")
                ],
                (574016,): [NumpyFeature(data=np.array([573.0]), name="costs")],
            },
            sampling=numpy_sampling,
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
        numpy_sampling = NumpySampling(
            data={
                # make index a datetime
                (666964,): np.array([1, 2]),
                (574016,): np.array([3]),
            },
            index=["product_id"],
        )

        numpy_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(data=np.array([740.0, 508.0]), name="costs")
                ],
                (574016,): [NumpyFeature(data=np.array([573.0]), name="costs")],
            },
            sampling=numpy_sampling,
        )

        expected_df = pd.DataFrame(
            [
                # use timestamp as datetime from numpy
                [666964, 740.0, 1],
                [666964, 508.0, 2],
                [574016, 573.0, 3],
            ],
            columns=["product_id", "costs", "timestamp"],
        )

        df = numpy_event.to_dataframe()

        # validate
        self.assertTrue(df.equals(expected_df))

    def test_numpy_event_to_df_no_index(self) -> None:
        numpy_sampling = NumpySampling(
            data={
                (): np.array([1.0, 2.0, 3.0]),
            },
            index=[],
        )

        numpy_event = NumpyEvent(
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
                ("X1", "Y1"): [
                    NumpyFeature(
                        name="sma_a",
                        data=np.array([10.0, 10.5, 11.0]),
                    )
                ],
                ("X2", "Y1"): [
                    NumpyFeature(
                        name="sma_a",
                        data=np.array([13.0, 13.5, 14.0]),
                    )
                ],
                ("X2", "Y2"): [
                    NumpyFeature(
                        name="sma_a",
                        data=np.array([16.0, 16.5, 17.0]),
                    )
                ],
            },
            sampling=NumpySampling(
                index=["x", "y"],
                data={
                    ("X1", "Y1"): np.array([1.0, 2.0, 3.0], dtype=np.float64),
                    ("X2", "Y1"): np.array([1.1, 2.1, 3.1], dtype=np.float64),
                    ("X2", "Y2"): np.array([1.2, 2.2, 3.2], dtype=np.float64),
                },
            ),
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


if __name__ == "__main__":
    absltest.main()
