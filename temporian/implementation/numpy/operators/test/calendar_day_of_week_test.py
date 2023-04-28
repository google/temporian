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

from temporian.core.operators.calendar.day_of_week import (
    CalendarDayOfWeekOperator,
)
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.calendar.day_of_week import (
    CalendarDayOfWeekNumpyImplementation,
)


class CalendarDayOfWeekNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of calendar_day_of_week operator."""

    def test_basic(self) -> None:
        "Basic test with flat node."
        input_evset = EventSet.from_dataframe(
            pd.DataFrame(
                data=[
                    [pd.to_datetime("Monday Mar 13 12:00:00 2023", utc=True)],
                    [pd.to_datetime("Tuesday Mar 14 12:00:00 2023", utc=True)],
                    [pd.to_datetime("Friday Mar 17 00:00:01 2023", utc=True)],
                    [pd.to_datetime("Friday Mar 17 23:59:59 2023", utc=True)],
                    [pd.to_datetime("Sunday Mar 19 23:59:59 2023", utc=True)],
                ],
                columns=["timestamp"],
            ),
        )
        input_node = input_evset.node()
        output_evset = EventSet(
            data={
                (): IndexData(
                    [np.array([0, 1, 4, 4, 6]).astype(np.int32)],
                    input_evset.first_index_data().timestamps,
                ),
            },
            feature_names=["calendar_day_of_week"],
            index_names=[],
            is_unix_timestamp=True,
        )
        operator = CalendarDayOfWeekOperator(input_node)
        impl = CalendarDayOfWeekNumpyImplementation(operator)
        output = impl.call(sampling=input_evset)

        self.assertTrue(output_evset == output["node"])
        self.assertTrue(
            output["node"].first_index_data().features[0].dtype == np.int32
        )


if __name__ == "__main__":
    absltest.main()
