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

"""Calendar ISO week operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import EventSetNode
from temporian.core.operators.calendar.base import BaseCalendarOperator
from temporian.core.typing import EventSetOrNode


class CalendarISOWeekOperator(BaseCalendarOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "CALENDAR_ISO_WEEK"

    @classmethod
    def output_feature_name(cls) -> str:
        return "calendar_iso_week"


operator_lib.register_operator(CalendarISOWeekOperator)


@compile
def calendar_iso_week(sampling: EventSetOrNode) -> EventSetOrNode:
    """Obtains the ISO week the timestamps in an
    [`EventSet`][temporian.EventSet]'s sampling are in.

    Features in `input` are ignored, only the timestamps are used and
    they must be unix timestamps (`is_unix_timestamp=True`).

    Output feature contains numbers between 1 and 53.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...    # Note: 2023-01-01 is Sunday in the same week as 2022-12-31
        ...    timestamps=["2022-12-31", "2023-01-01", "2023-01-02", "2023-12-20"],
        ... )
        >>> b = tp.calendar_iso_week(a)
        >>> b
        indexes: ...
        features: [('calendar_iso_week', int32)]
        events:
            (4 events):
                timestamps: [...]
                'calendar_iso_week': [52 52 1 51]
        ...

        ```
    Args:
        sampling: EventSet to get the ISO weeks from.

    Returns:
        EventSet with a single feature with the ISO week each timestamp in
        `sampling` belongs to.
    """
    assert isinstance(sampling, EventSetNode)

    return CalendarISOWeekOperator(sampling).outputs["output"]
