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
from absl import logging
from absl.testing import absltest
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


class EventConversionTest(absltest.TestCase):
    def test_df_to_numpy_event(self) -> None:
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
            df, index_names=["product_id"], timestamp_name="timestamp"
        )

        # validate
        self.assertEqual(
            True,
            numpy_event == expected_numpy_event,
        )

    def test_df_to_numpy_event_no_index(self) -> None:
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
            df, index_names=[], timestamp_name="timestamp"
        )

        # validate
        self.assertEqual(
            True,
            numpy_event == expected_numpy_event,
        )

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
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        ).set_index(["product_id", "timestamp"])

        df = numpy_event.to_dataframe()

        # validate
        self.assertEqual(
            True,
            df.equals(expected_df),
        )

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
                [666964, 1.0, 740.0],
                [666964, 2.0, 508.0],
                [574016, 3.0, 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        ).set_index(["timestamp"])

        df = numpy_event.to_dataframe()

        # validate
        self.assertEqual(
            True,
            df.equals(expected_df),
        )


if __name__ == "__main__":
    absltest.main()
