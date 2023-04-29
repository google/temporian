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

import math
from absl.testing import absltest

import pandas as pd

from temporian.core.operators.since_last import SinceLast
from temporian.implementation.numpy.operators.since_last import (
    SinceLastNumpyImplementation,
    operators_cc,
)
from temporian.implementation.numpy.data.event import NumpyEvent

nan = math.nan


class SinceLastOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_no_sampling(self):
        event_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 5, 8, 9, 1, 1, 2],
                    "x": [1, 1, 1, 1, 2, 2, 2],
                }
            ),
            index_names=["x"],
        )
        event = event_data.schema()

        expected_output = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 5, 8, 9, 1, 1, 2],
                    "x": [1, 1, 1, 1, 2, 2, 2],
                    "since_last": [nan, 4, 3, 1, nan, 0, 1],
                }
            ),
            index_names=["x"],
        )

        # Run op
        op = SinceLast(event=event)
        instance = SinceLastNumpyImplementation(op)
        output = instance.call(event=event_data)["event"]

        self.assertEqual(output, expected_output)

    def test_with_sampling(self):
        event_data = NumpyEvent.from_dataframe(
            pd.DataFrame({"timestamp": [1, 2, 2, 4]}),
        )
        event = event_data.schema()

        sampling_data = NumpyEvent.from_dataframe(
            pd.DataFrame({"timestamp": [-1, 1, 1.5, 2, 2.1, 4, 5]})
        )
        sampling = sampling_data.schema()

        expected_output = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [-1, 1, 1.5, 2, 2.1, 4, 5],
                    "since_last": [nan, 0, 0.5, 0, 0.1, 0, 1],
                }
            )
        )

        # Run op
        op = SinceLast(event=event, sampling=sampling)
        instance = SinceLastNumpyImplementation(op)
        output = instance.call(event=event_data, sampling=sampling_data)[
            "event"
        ]

        print("@output:\n", output)
        print("@expected_output:\n", expected_output)

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
