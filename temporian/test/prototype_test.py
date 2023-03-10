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
from absl import logging
from absl.testing import absltest
from temporian.core import evaluator
from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.assign import assign
from temporian.core.operators.sum import sum
from temporian.core.operators.simple_moving_average import sma
from temporian.implementation.pandas.data import event as pandas_event


class PrototypeTest(absltest.TestCase):
    def setUp(self) -> None:
        self.left_event = "temporian/test/test_data/prototype/left_event.csv"

        self.right_event = pandas_event.PandasEvent(
            [
                [666964, pd.Timestamp("2013-01-02"), 740.0],
                [666964, pd.Timestamp("2013-01-03"), 508.0],
                [574016, pd.Timestamp("2013-01-04"), 573.0],
            ],
            columns=["product_id", "timestamp", "costs"],
        ).set_index(["product_id", "timestamp"])

        self.expected_output_event = pandas_event.PandasEvent(
            [
                [
                    666964,
                    pd.Timestamp("2013-01-02"),
                    1091.0,
                    740.0,
                    740.0,
                    1831.0,
                ],
                [
                    666964,
                    pd.Timestamp("2013-01-03"),
                    919.0,
                    508.0,
                    624.0,
                    1427.0,
                ],
                [
                    574016,
                    pd.Timestamp("2013-01-04"),
                    953.0,
                    573.0,
                    573.0,
                    1526.0,
                ],
            ],
            columns=[
                "product_id",
                "timestamp",
                "sales",
                "costs",
                "sma_costs",
                "sum_sales_costs",
            ],
        ).set_index(["product_id", "timestamp"])

    def test_prototoype(self) -> None:
        # instance input events
        sampling = Sampling(["product_id", "timestamp"])

        left_event = Event(
            features=[Feature(name="sales", dtype=float)],
            sampling=sampling,
            creator=None,
        )
        right_event = Event(
            features=[Feature(name="costs", dtype=float)],
            sampling=sampling,
            creator=None,
        )

        sum_events = sum(
            event_1=left_event,
            event_2=right_event,
        )

        # call assign operator
        output_event = assign(left_event, right_event)

        # call sma operator
        sma_right_event = sma(
            right_event, window_length="7d", sampling=right_event
        )

        # call assign operator with result of sma
        output_event = assign(output_event, sma_right_event)

        output_event = assign(output_event, sum_events)

        # evaluate output
        output_event_pandas = evaluator.evaluate(
            output_event,
            input_data={
                # left event specified from disk
                left_event: self.left_event,
                # right event loaded in ram
                right_event: self.right_event,
            },
            backend="pandas",
        )

        # validate
        self.assertEqual(
            True,
            self.expected_output_event.equals(
                output_event_pandas[output_event]
            ),
        )


if __name__ == "__main__":
    absltest.main()
