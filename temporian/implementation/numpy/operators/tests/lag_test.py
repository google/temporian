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

import pandas as pd

from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.lag import LagOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators import lag


class LagOperatorTest(absltest.TestCase):
    """Lag operator test."""

    def test_correct_lag(self) -> None:
        """Test correct lag operator."""
        numpy_input_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1, 10.0],
                    [0, 1.5, 11.0],
                    [0, 3, 12.0],
                    [0, 3.5, 13.0],
                    [0, 4, 14.0],
                    [0, 10, 15.0],
                    [0, 20, 16.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 3, 10.0],
                    [0, 3.5, 11.0],
                    [0, 5, 12.0],
                    [0, 5.5, 13.0],
                    [0, 6, 14.0],
                    [0, 12, 15.0],
                    [0, 22, 16.0],
                ],
                columns=["store_id", "timestamp", "lag[2s]_sales"],
            ),
            index_names=["store_id"],
        )

        event = Event(
            [Feature("sales", float)],
            sampling=Sampling(["store_id"]),
            creator=None,
        )

        operator = LagOperator(
            duration=2,
            event=event,
        )

        lag_implementation = lag.LagNumpyImplementation(operator)
        operator_output = lag_implementation(event=numpy_input_event)

        self.assertTrue(numpy_output_event == operator_output["event"])

    def test_correct_leak(self) -> None:
        """Test correct leak operator."""
        numpy_input_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1, 10.0],
                    [0, 1.5, 11.0],
                    [0, 3, 12.0],
                    [0, 3.5, 13.0],
                    [0, 4, 14.0],
                    [0, 10, 15.0],
                    [0, 20, 16.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, -1, 10.0],
                    [0, -0.5, 11.0],
                    [0, 1, 12.0],
                    [0, 1.5, 13.0],
                    [0, 2, 14.0],
                    [0, 8, 15.0],
                    [0, 18, 16.0],
                ],
                columns=["store_id", "timestamp", "leak[2s]_sales"],
            ),
            index_names=["store_id"],
        )

        event = Event(
            [Feature("sales", float)],
            sampling=Sampling(["store_id"]),
            creator=None,
        )

        operator = LagOperator(
            duration=-2,
            event=event,
        )

        lag_implementation = lag.LagNumpyImplementation(operator)
        operator_output = lag_implementation(event=numpy_input_event)

        self.assertTrue(numpy_output_event == operator_output["event"])


if __name__ == "__main__":
    absltest.main()
