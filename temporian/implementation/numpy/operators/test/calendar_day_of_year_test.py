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

from temporian.core.operators.calendar.day_of_year import (
    CalendarDayOfYearOperator,
)
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.calendar.day_of_year import (
    CalendarDayOfYearNumpyImplementation,
)


class CalendarDayOfYearNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of calendar_day_of_year operator."""

    def test_basic(self) -> None:
        "Basic test with flat node."
        input_evset = EventSet.from_dataframe(
            pd.DataFrame(
                data=[
                    [pd.to_datetime("1970-01-01 00:00:00", utc=True)],
                    [pd.to_datetime("1970-01-02 00:00:00", utc=True)],
                    [pd.to_datetime("2023-01-15 23:59:59", utc=True)],
                    [pd.to_datetime("2023-06-15 15:30:00", utc=True)],
                    [pd.to_datetime("2023-12-31 12:00:00", utc=True)],
                    [
                        pd.to_datetime("2024-12-31 23:59:59", utc=True)
                    ],  # 2024 is a leap year, so this is day 366
                ],
                columns=["timestamp"],
            ),
        )
        input_node = input_evset.node()
        output_evset = EventSet(
            data={
                (): IndexData(
                    [np.array([1, 2, 15, 166, 365, 366]).astype(np.int32)],
                    input_evset.first_index_data().timestamps,
                ),
            },
            feature_names=["calendar_day_of_year"],
            index_names=[],
            is_unix_timestamp=True,
        )
        operator = CalendarDayOfYearOperator(input_node)
        impl = CalendarDayOfYearNumpyImplementation(operator)
        output = impl.call(sampling=input_evset)

        self.assertTrue(output_evset == output["node"])
        self.assertTrue(
            output["node"].first_index_data().features[0].dtype == np.int32
        )


if __name__ == "__main__":
    absltest.main()
