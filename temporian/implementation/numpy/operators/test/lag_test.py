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
        expected_lag_1_numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 2.0, 10.0],
                    [0, 3.0, 11.0],
                    [0, 4.0, 12.0],
                    [0, 5.0, 13.0],
                    [0, 6.0, 14.0],
                    [0, 7.0, 15.0],
                    [0, 8.0, 16.0],
                ],
                columns=[
                    "store_id",
                    "timestamp",
                    "lag[1s]_sales",
                ],
            ),
            index_names=["store_id"],
        )

        expected_lag_2_numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 3.0, 10.0],
                    [0, 4.0, 11.0],
                    [0, 5.0, 12.0],
                    [0, 6.0, 13.0],
                    [0, 7.0, 14.0],
                    [0, 8.0, 15.0],
                    [0, 9.0, 16.0],
                ],
                columns=[
                    "store_id",
                    "timestamp",
                    "lag[2s]_sales",
                ],
            ),
            index_names=["store_id"],
        )

        event = numpy_input_event.schema()

        # lag multiple durations
        lags = lag(event=event, duration=[1, 2])

        lag_1 = lags[0]

        # evaluate
        output_event_numpy_lag_1 = evaluator.evaluate(
            lag_1,
            input_data={
                event: numpy_input_event,
            },
        )

        # validate
        self.assertEqual(
            expected_lag_1_numpy_output_event, output_event_numpy_lag_1
        )

        lag_2 = lags[1]

        # evaluate
        output_event_numpy_lag_2 = evaluator.evaluate(
            lag_2,
            input_data={
                event: numpy_input_event,
            },
        )

        # validate
        self.assertEqual(
            expected_lag_2_numpy_output_event, output_event_numpy_lag_2
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
        expected_leak_1_numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, 0.0, 10.0],
                    [0, 1.0, 11.0],
                    [0, 2.0, 12.0],
                    [0, 3.0, 13.0],
                    [0, 4.0, 14.0],
                    [0, 5.0, 15.0],
                    [0, 6.0, 16.0],
                ],
                columns=[
                    "store_id",
                    "timestamp",
                    "leak[1s]_sales",
                ],
            ),
            index_names=["store_id"],
        )

        expected_leak_2_numpy_output_event = NumpyEvent.from_dataframe(
            pd.DataFrame(
                [
                    [0, -1.0, 10.0],
                    [0, 0.0, 11.0],
                    [0, 1.0, 12.0],
                    [0, 2.0, 13.0],
                    [0, 3.0, 14.0],
                    [0, 4.0, 15.0],
                    [0, 5.0, 16.0],
                ],
                columns=[
                    "store_id",
                    "timestamp",
                    "leak[2s]_sales",
                ],
            ),
            index_names=["store_id"],
        )

        event = numpy_input_event.schema()

        # leak multiple durations
        leaks = leak(event=event, duration=[1, 2])

        leak_1 = leaks[0]

        # evaluate
        output_event_numpy_leak_1 = evaluator.evaluate(
            leak_1,
            input_data={
                event: numpy_input_event,
            },
        )

        # validate
        self.assertEqual(
            expected_leak_1_numpy_output_event, output_event_numpy_leak_1
        )

        leak_2 = leaks[1]

        # evaluate
        output_event_numpy_leak_2 = evaluator.evaluate(
            leak_2,
            input_data={
                event: numpy_input_event,
            },
        )

        # validate
        self.assertEqual(
            expected_leak_2_numpy_output_event, output_event_numpy_leak_2
        )


if __name__ == "__main__":
    absltest.main()
