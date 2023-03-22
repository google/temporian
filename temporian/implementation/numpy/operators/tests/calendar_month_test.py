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

from temporian.core.operators.calendar.month import (
    CalendarMonthOperator,
)
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.operators.calendar.month import (
    CalendarMonthNumpyImplementation,
)


class CalendarMonthNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of calendar_month operator."""

    def test_basic(self) -> None:
        "Basic test with flat event."
        input_event_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                data=[
                    [pd.to_datetime("1970-01-01 00:00:00", utc=True)],
                    [pd.to_datetime("2021-01-01 00:00:00", utc=True)],
                    [pd.to_datetime("2021-07-15 12:30:00", utc=True)],
                    [pd.to_datetime("2021-12-31 23:59:59", utc=True)],
                    [pd.to_datetime("2045-12-31 23:59:59", utc=True)],
                    [pd.to_datetime("2045-12-01 00:00:00", utc=True)],
                ],
                columns=["timestamp"],
            ),
        )

        input_event = input_event_data.schema()

        output_event_data = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        name="calendar_month",
                        data=np.array([1, 1, 7, 12, 12, 12]),
                    ),
                ],
            },
            sampling=input_event_data.sampling,
        )

        operator = CalendarMonthOperator(input_event)
        impl = CalendarMonthNumpyImplementation(operator)

        output = impl(input_event_data)

        self.assertTrue(output_event_data == output["event"])
        self.assertTrue(
            output["event"]._first_index_features[0].dtype == np.int32
        )


if __name__ == "__main__":
    absltest.main()
