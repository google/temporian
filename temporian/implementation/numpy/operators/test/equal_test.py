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

from temporian.core.operators.equal import EqualOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators import equal


class EqualOperatorTest(absltest.TestCase):
    """Equal operator test."""

    def setUp(self):
        df = pd.DataFrame(
            [
                [1.0, 10.0, "A", 10],
                [2.0, np.nan, "A", 12],
                [3.0, 12.0, "B", 40],
                [4.0, 13.0, "B", 100],
                [5.0, 14.0, "10.0", 0],
                [6.0, 15.0, "10", 15],
            ],
            columns=["timestamp", "sales", "costs", "weather"],
        )

        df["weather"] = df["weather"].astype(np.int64)

        self.input_event_data = NumpyEvent.from_dataframe(df)

        self.input_event = self.input_event_data.schema()

    def test_float_equal(self) -> None:
        """Test equal when value is a float."""
        new_df = pd.DataFrame(
            [
                [1.0, True, False, True],
                [2.0, False, False, False],
                [3.0, False, False, False],
                [4.0, False, False, False],
                [5.0, False, False, False],
                [6.0, False, False, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_10.0",
                "costs_equal_10.0",
                "weather_equal_10.0",
            ],
        )

        operator = EqualOperator(event=self.input_event, value=10.0)
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event=self.input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)

    def test_string_equal(self) -> None:
        """Test equal when value is a string."""
        new_df = pd.DataFrame(
            [
                [1.0, False, False, False],
                [2.0, False, False, False],
                [3.0, False, False, False],
                [4.0, False, False, False],
                [5.0, False, False, False],
                [6.0, False, True, False],
            ],
            columns=[
                "timestamp",
                "sales_equal_10",
                "costs_equal_10",
                "weather_equal_10",
            ],
        )

        operator = EqualOperator(event=self.input_event, value="10")
        impl = equal.EqualNumpyImplementation(operator)
        equal_event = impl.call(event=self.input_event_data)["event"]

        expected_event = NumpyEvent.from_dataframe(new_df)

        self.assertEqual(equal_event, expected_event)


if __name__ == "__main__":
    absltest.main()
