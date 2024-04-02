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

from typing import Dict, Literal, Tuple, Union

import numpy as np

from temporian.core.operators.tick_calendar import TickCalendar
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet, IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.implementation.numpy_cc.operators import operators_cc


class TickCalendarNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: TickCalendar) -> None:
        assert isinstance(operator, TickCalendar)
        super().__init__(operator)

    def _wday_py_to_cpp(self, py_wday: int) -> int:
        """Converts wday number from Python (wday=0 for Monday) to C++
        convention (wday=0 for Sunday).
        This is required to keep coherency between calendar_day_of_week()
        operator (which uses datetime.weekday()) and tick_calendar() (which
        uses tm_wday in tick_calendar.cc)."""
        return (py_wday + 1) % 7

    def _get_arg_range(
        self,
        arg_value: Union[int, Literal["*"]],
        val_range: Tuple[int, int],
    ):
        if arg_value == "*":
            range_ini, range_end = val_range
        else:
            range_ini = range_end = arg_value

        return range_ini, range_end

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, TickCalendar)
        output_schema = self.output_schema("output")

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        # Get range for each argument
        second_range = self._get_arg_range(
            self.operator.second, self.operator.seconds_max_range()
        )
        minute_range = self._get_arg_range(
            self.operator.minute, self.operator.minutes_max_range()
        )
        hour_range = self._get_arg_range(
            self.operator.hour, self.operator.hours_max_range()
        )
        mday_range = self._get_arg_range(
            self.operator.mday, self.operator.mday_max_range()
        )
        month_range = self._get_arg_range(
            self.operator.month, self.operator.month_max_range()
        )

        # Weekday: convert python (wday=0 for Mon) to C++ (wday=0 for Sun)
        wday = self.operator.wday
        if wday != "*":
            wday = self._wday_py_to_cpp(wday)
        wday_range = self._get_arg_range(wday, self.operator.wday_max_range())

        after_last = self.operator.after_last
        before_first = self.operator.before_first

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            if len(index_data.timestamps) == 0:
                dst_timestamps = np.array([], dtype=np.float64)
            else:
                dst_timestamps = operators_cc.tick_calendar(
                    start_timestamp=index_data.timestamps[0],
                    end_timestamp=index_data.timestamps[-1],
                    min_second=second_range[0],
                    max_second=second_range[1],
                    min_minute=minute_range[0],
                    max_minute=minute_range[1],
                    min_hour=hour_range[0],
                    max_hour=hour_range[1],
                    min_mday=mday_range[0],
                    max_mday=mday_range[1],
                    min_month=month_range[0],
                    max_month=month_range[1],
                    min_wday=wday_range[0],
                    max_wday=wday_range[1],
                    after_last=after_last,
                    before_first=before_first,
                )
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
