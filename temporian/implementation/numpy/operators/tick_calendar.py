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


"""Implementation for the TickCalendar operator."""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.tick_calendar import TickCalendar
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.core.data import duration


class TickCalendarNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: TickCalendar) -> None:
        assert isinstance(operator, TickCalendar)
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, TickCalendar)
        second = self.operator.second
        minute = self.operator.minute
        hour = self.operator.hour
        day_of_month = self.operator.day_of_month
        month = self.operator.month
        day_of_week = self.operator.day_of_week

        output_schema = self.output_schema("output")

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            if len(index_data.timestamps < 2):
                dst_timestamps = np.array([], dtype=np.float64)
            else:
                begin = index_data.timestamps[0]
                end = index_data.timestamps[-1]
            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=[],
                    timestamps=dst_timestamps,
                    schema=output_schema,
                ),
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    TickCalendar, TickCalendarNumpyImplementation
)
