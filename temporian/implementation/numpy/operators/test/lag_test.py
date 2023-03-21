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

from temporian.core import evaluator
from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.assign import assign
from temporian.core.operators.lag import lag
from temporian.core.operators.lag import leak
from temporian.core.operators.lag import LagOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.lag import LagNumpyImplementation


class LagNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of lag operator."""

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

        lag_implementation = LagNumpyImplementation(operator)
        operator_output = lag_implementation(event=numpy_input_event)

        self.assertTrue(numpy_output_event == operator_output["event"])

    def test_correct_multiple_lags(self) -> None:
        """Test correct lag operator with duration list."""
        numpy_input_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0],
                    [0, 2.0, 11.0],
                    [0, 3.0, 12.0],
                    [0, 4.0, 13.0],
                    [0, 5.0, 14.0],
                    [0, 6.0, 15.0],
                    [0, 7.0, 16.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        expected_numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0, np.nan, np.nan],
                    [0, 2.0, 11.0, 10.0, np.nan],
                    [0, 3.0, 12.0, 11.0, 10.0],
                    [0, 4.0, 13.0, 12.0, 11.0],
                    [0, 5.0, 14.0, 13.0, 12.0],
                    [0, 6.0, 15.0, 14.0, 13.0],
                    [0, 7.0, 16.0, 15.0, 14.0],
                ],
                columns=[
                    "store_id",
                    "timestamp",
                    "sales",
                    "lag[1s]_sales",
                    "lag[2s]_sales",
                ],
            ),
            index_names=["store_id"],
        )

        event = numpy_input_event.schema()

        # lag multiple durations
        lags = lag(event=event, duration=[1, 2])

        # assign multiple lags to output event
        output_event = event
        for lagged_event in lags:
            output_event = assign(output_event, lagged_event)

        # evaluate
        output_event_numpy = evaluator.evaluate(
            output_event,
            input_data={
                event: numpy_input_event,
            },
            backend="numpy",
        )

        # validate
        self.assertEqual(
            expected_numpy_output_event, output_event_numpy[output_event]
        )

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

        lag_implementation = LagNumpyImplementation(operator)
        operator_output = lag_implementation(event=numpy_input_event)

        self.assertTrue(numpy_output_event == operator_output["event"])

    def test_correct_multiple_leaks(self) -> None:
        """Test correct leak operator with duration list."""
        numpy_input_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0],
                    [0, 2.0, 11.0],
                    [0, 3.0, 12.0],
                    [0, 4.0, 13.0],
                    [0, 5.0, 14.0],
                    [0, 6.0, 15.0],
                    [0, 7.0, 16.0],
                ],
                columns=["store_id", "timestamp", "sales"],
            ),
            index_names=["store_id"],
        )

        expected_numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 1.0, 10.0, 11.0, 12.0],
                    [0, 2.0, 11.0, 12.0, 13.0],
                    [0, 3.0, 12.0, 13.0, 14.0],
                    [0, 4.0, 13.0, 14.0, 15.0],
                    [0, 5.0, 14.0, 15.0, 16.0],
                    [0, 6.0, 15.0, 16.0, np.nan],
                    [0, 7.0, 16.0, np.nan, np.nan],
                ],
                columns=[
                    "store_id",
                    "timestamp",
                    "sales",
                    "leak[1s]_sales",
                    "leak[2s]_sales",
                ],
            ),
            index_names=["store_id"],
        )

        event = numpy_input_event.schema()

        # lag multiple durations
        leaks = leak(event=event, duration=[1, 2])

        # assign multiple lags to output event
        output_event = event
        for leaked_event in leaks:
            output_event = assign(output_event, leaked_event)

        # evaluate
        output_event_numpy = evaluator.evaluate(
            output_event,
            input_data={
                event: numpy_input_event,
            },
            backend="numpy",
        )

        # validate
        self.assertEqual(
            expected_numpy_output_event, output_event_numpy[output_event]
        )


if __name__ == "__main__":
    absltest.main()
