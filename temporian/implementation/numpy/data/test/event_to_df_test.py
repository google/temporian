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
from absl.testing import absltest
from temporian.implementation.numpy.data.io import (
    event_set,
    event_set_to_pd_dataframe,
)


class EventToDataFrameTest(absltest.TestCase):
    def test_evset_to_df(self) -> None:
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": [740.0, 508.0, 573.0],
            },
            index_features=["product_id"],
        )

        expected_df = pd.DataFrame(
            [
                [666964, 740.0, 1.0],
                [666964, 508.0, 2.0],
                [574016, 573.0, 3.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )
        df = event_set_to_pd_dataframe(evset)

        self.assertTrue(df.equals(expected_df))

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
        df = event_set_to_pd_dataframe(evset)

        self.assertTrue(df.equals(expected_df))

    def test_evset_to_df_multiple_index(self) -> None:
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            features={
                "sma_a": [10.0, 10.5, 11.0, 13.0, 13.5, 14.0, 16.0, 16.5, 17.0],
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
            },
            index_features=["x", "y"],
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
        df = event_set_to_pd_dataframe(evset)

        self.assertTrue(df.equals(expected_df))

    def test_evset_to_df_string_feature(self) -> None:
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={
                "product_id": [666964, 666964, 574016],
                "costs": ["740.0", "508.0", "573.0"],
            },
            index_features=["product_id"],
        )
        expected_df = pd.DataFrame(
            [
                [666964, "740.0", 1.0],
                [666964, "508.0", 2.0],
                [574016, "573.0", 3.0],
            ],
            columns=["product_id", "costs", "timestamp"],
        )
        df = event_set_to_pd_dataframe(evset)

        self.assertTrue(df.equals(expected_df))


if __name__ == "__main__":
    absltest.main()
