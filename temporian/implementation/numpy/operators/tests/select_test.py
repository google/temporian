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

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators import select


class SelectOperatorTest(absltest.TestCase):
    """Select operator test."""

    def setUp(self):
        self.A = 0
        self.B = 1
        self.C = 2

        self.df = pd.DataFrame(
            [
                [self.A, 1, 10.0, -1.0, 0.0],
                [self.A, 2, np.nan, -2.0, 32.0],
                [self.B, 3, 12.0, -3.0, 27.0],
                [self.B, 4, 13.0, -4.0, 28.0],
                [self.C, 5, 14.0, np.nan, 29.0],
                [self.C, 6, 15.0, -6.0, np.nan],
            ],
            columns=["store_id", "timestamp", "sales", "costs", "weather"],
        )

        self.features = ["sales", "costs", "weather"]

        self.numpy_event = NumpyEvent.from_dataframe(
            self.df, index_names=["store_id"]
        )

    def test_select_one_feature(self) -> None:
        """Test correct select operator for one feature selection."""
        new_df = pd.DataFrame(
            [
                [self.A, 1, 10.0],
                [self.A, 2, np.nan],
                [self.B, 3, 12.0],
                [self.B, 4, 13.0],
                [self.C, 5, 14.0],
                [self.C, 6, 15.0],
            ],
            columns=["store_id", "timestamp", "sales"],
        )

        operator = select.NumpySelectOperator(feature_names="sales")
        selected_event = operator(self.numpy_event)["event"]

        expected_event = NumpyEvent.from_dataframe(
            new_df, index_names=["store_id"]
        )

        self.assertTrue(selected_event == expected_event)

    def test_select_multiple_features(self) -> None:
        """Test correct select operator for multiple features selection."""
        new_df = pd.DataFrame(
            [
                [self.A, 1, 10.0, -1.0],
                [self.A, 2, np.nan, -2.0],
                [self.B, 3, 12.0, -3.0],
                [self.B, 4, 13.0, -4.0],
                [self.C, 5, 14.0, np.nan],
                [self.C, 6, 15.0, -6.0],
            ],
            columns=["store_id", "timestamp", "sales", "costs"],
        )

        operator = select.NumpySelectOperator(feature_names=["sales", "costs"])
        selected_event = operator(self.numpy_event)["event"]

        expected_event = NumpyEvent.from_dataframe(
            new_df, index_names=["store_id"]
        )

        self.assertTrue(selected_event == expected_event)


if __name__ == "__main__":
    absltest.main()
