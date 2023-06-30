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

"""Calendar day of week operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.operators.base import EventSetOrNode
from temporian.core.operators.calendar.base import BaseCalendarOperator


class CalendarDayOfWeekOperator(BaseCalendarOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "CALENDAR_DAY_OF_WEEK"

    @classmethod
    def output_feature_name(cls) -> str:
        return "calendar_day_of_week"


operator_lib.register_operator(CalendarDayOfWeekOperator)


@compile
def calendar_day_of_week(sampling: EventSetOrNode) -> EventSetOrNode:
    """Obtains the day of the week the timestamps in a node's sampling are in.

    Features in the input node are ignored, only the timestamps are used and
    they must be unix timestamps (`is_unix_timestamp=True`).

    Output feature contains numbers from 0 (Monday) to 6 (Sunday).

    Usage example:
        ```python
        >>> evset = tp.event_set(
        ...    timestamps=["2023-06-19", "2023-06-21", "2023-06-25", "2023-07-03"],
        ...    name='two_mondays'
        ... )
        >>> tp.calendar_day_of_week(evset.node()).run(evset)
        indexes: ...
        features: [('calendar_day_of_week', int32)]
        events:
            (4 events):
                timestamps: [...]
                'calendar_day_of_week': [0  2  6  0]
        ...

        ```

    Args:
        sampling: Node to get the days of week from.

    Returns:
        Single feature with the day each timestamp in `sampling` belongs to.
    """
    return CalendarDayOfWeekOperator(sampling).outputs["output"]
