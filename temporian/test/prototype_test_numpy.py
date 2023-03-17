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
from absl import logging
from absl.testing import absltest

import numpy as np
import pandas as pd

from temporian.core import evaluator
from temporian.core.data.event import Event, Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.assign import assign
from temporian.core.operators.lag import lag
from temporian.implementation.numpy.data.event import NumpyEvent


class PrototypeTest(absltest.TestCase):
    def setUp(self) -> None:
        # store ids
        TRYOLABS_SHOP = 42
        GOOGLE_SHOP = 101
        # product ids
        MATE_ID = 1
        BOOK_ID = 2
        PIXEL_ID = 3

        self.event_1_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                data=[
                    [TRYOLABS_SHOP, MATE_ID, 0.0, 14],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, 15],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, 16],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, 10],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 7],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, 8],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, 3],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, 4],
                ],
                columns=["store_id", "product_id", "timestamp", "sales"],
            ),
            index_names=["store_id", "product_id"],
        )

        self.event_2_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                data=[
                    [TRYOLABS_SHOP, MATE_ID, 0.0, -14],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, -15],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, -16],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, -10],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, -7],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, -8],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, -3],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, -4],
                ],
                columns=["store_id", "product_id", "timestamp", "costs"],
            ),
            index_names=["store_id", "product_id"],
        )

        self.expected_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                data=[
                    [TRYOLABS_SHOP, MATE_ID, 0.0, 14, -14, 0, np.nan],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, 15, -15, 0, 14],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, 16, -16, 0, 15],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, 10, -10, 0, np.nan],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 7, -7, 0, np.nan],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, 8, -8, 0, 7],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, 3, -3, 0, np.nan],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, 4, -4, 0, 3],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "sales",
                    "costs",
                    "add_sales_costs",
                    "lag[1s]_sales",
                ],
            ),
            index_names=["store_id", "product_id"],
        )

    def test_prototype(self) -> None:
        event_1 = self.event_1_data.schema()
        event_2 = self.event_2_data.schema()

        lagged_sales = lag(
            event_1,
            duration=1,
        )

        # add costs feature to output
        output_event = assign(event_1, event_2)

        # add sum of sales and costs
        output_event = assign(output_event, event_1 + event_2)

        output_event = assign(output_event, lagged_sales)

        output_event_numpy = evaluator.evaluate(
            output_event,
            input_data=[event_1, event_2],
            backend="numpy",
        )

        if self.expected_output_event != output_event_numpy[output_event]:
            print("expected")
            print(self.expected_output_event.to_dataframe())
            print("actual")
            print(output_event_numpy[output_event].to_dataframe())

        # validate
        self.assertEqual(
            self.expected_output_event, output_event_numpy[output_event]
        )


if __name__ == "__main__":
    absltest.main()
