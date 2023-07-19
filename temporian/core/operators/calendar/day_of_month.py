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

"""Calendar day of month operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import EventSetNode
from temporian.core.operators.calendar.base import BaseCalendarOperator
from temporian.core.typing import EventSetOrNode


class CalendarDayOfMonthOperator(BaseCalendarOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "CALENDAR_DAY_OF_MONTH"

    @classmethod
    def output_feature_name(cls) -> str:
        return "calendar_day_of_month"


operator_lib.register_operator(CalendarDayOfMonthOperator)


@compile
def calendar_day_of_month(sampling: EventSetOrNode) -> EventSetOrNode:
    """Obtains the day of month the timestamps in an
    [`EventSet`][temporian.EventSet]'s sampling are in.

    Features in `input` are ignored, only the timestamps are used and
    they must be unix timestamps (`is_unix_timestamp=True`).

    Output feature contains numbers between 1 and 31.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...    timestamps=["2023-02-04", "2023-02-20", "2023-03-01", "2023-05-07"],
        ... )
        >>> b = tp.calendar_day_of_month(a)
        >>> b
        indexes: ...
        features: [('calendar_day_of_month', int32)]
        events:
            (4 events):
                timestamps: [...]
                'calendar_day_of_month': [ 4 20  1  7]
        ...

        ```

    Args:
        sampling: EventSet to get the days of month from.

    Returns:
        EventSet with a single feature with the day of the month each timestamp
        in `sampling` belongs to.
    """
    assert isinstance(sampling, EventSetNode)

    return CalendarDayOfMonthOperator(sampling).outputs["output"]
