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
import numpy as np
import pandas as pd
from absl import logging
from absl.testing import absltest

from temporian.core import evaluator
from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.assign import assign
from temporian.implementation.numpy.data.event import NumpyEvent


class PrototypeTest(absltest.TestCase):
    def setUp(self) -> None:
        self.event_1 = NumpyEvent.from_dataframe(
            pd.DataFrame(
                data=[
                    ["A", 1, "2022-02-05", 14],
                    ["A", 1, "2022-02-06", 15],
                    ["A", 1, "2022-02-07", 16],
                    ["A", 2, "2022-02-06", 10],
                    ["B", 2, "2022-02-06", 7],
                    ["B", 2, "2022-02-07", 8],
                    ["B", 3, "2022-02-05", 3],
                    ["B", 3, "2022-02-06", 4],
                ],
                columns=["store_id", "product_id", "timestamp", "sales"],
            ),
            index_names=["store_id", "product_id", "timestamp"],
        )

        self.event_2 = NumpyEvent.from_dataframe(
            pd.DataFrame(
                data=[
                    ["A", 1, "2022-02-05", -14],
                    ["A", 1, "2022-02-06", -15],
                    ["A", 1, "2022-02-07", -16],
                    ["A", 2, "2022-02-06", -10],
                    ["B", 2, "2022-02-06", -7],
                    ["B", 2, "2022-02-07", -8],
                    ["B", 3, "2022-02-05", -3],
                    ["B", 3, "2022-02-06", -4],
                ],
                columns=["store_id", "product_id", "timestamp", "costs"],
            ),
            index_names=["store_id", "product_id", "timestamp"],
        )

        self.expected_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                data=[
                    ["A", 1, "2022-02-05", 14, -14, 0],
                    ["A", 1, "2022-02-06", 15, -15, 0],
                    ["A", 1, "2022-02-07", 16, -16, 0],
                    ["A", 2, "2022-02-06", 10, -10, 0],
                    ["B", 2, "2022-02-06", 7, -7, 0],
                    ["B", 2, "2022-02-07", 8, -8, 0],
                    ["B", 3, "2022-02-05", 3, -3, 0],
                    ["B", 3, "2022-02-06", 4, -4, 0],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "sales",
                    "costs",
                    "add_sales_costs",
                ],
            ),
            index_names=["store_id", "product_id", "timestamp"],
        )

    def test_prototype(self) -> None:
        sampling = Sampling(["store_id", "product_id"])
        event_1 = Event(
            [Feature("sales", int)],
            sampling=sampling,
            creator=None,
        )

        event_2 = Event(
            [Feature("costs", int)],
            sampling=sampling,
        )

        # add costs feature to output
        output_event = assign(event_1, event_2)

        # add sum of sales and costs
        output_event = assign(output_event, event_1 + event_2)

        output_event_numpy = evaluator.evaluate(
            output_event,
            input_data={
                # left event specified from disk
                event_1: self.event_1,
                event_2: self.event_2,
            },
            backend="numpy",
        )

        # validate
        self.assertEqual(
            True,
            self.expected_output_event == output_event_numpy[output_event],
        )


if __name__ == "__main__":
    absltest.main()
