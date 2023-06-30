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

"""Calendar month operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.operators.base import EventSetOrNode
from temporian.core.operators.calendar.base import BaseCalendarOperator


class CalendarMonthOperator(BaseCalendarOperator):
    """
    Calendar operator to obtain the month each timestamp belongs to.
    """

    @classmethod
    def operator_def_key(cls) -> str:
        return "CALENDAR_MONTH"

    @classmethod
    def output_feature_name(cls) -> str:
        return "calendar_month"


operator_lib.register_operator(CalendarMonthOperator)


@compile
def calendar_month(sampling: EventSetOrNode) -> EventSetOrNode:
    """Obtains the month the timestamps in a node's sampling are in.

    Features in the input node are ignored, only the timestamps are used and
    they must be unix timestamps (`is_unix_timestamp=True`).

    Output feature contains numbers between
    1 and 12.

    Usage example:
        ```python
        >>> evset = tp.event_set(
        ...    timestamps=["2023-02-04", "2023-02-20", "2023-03-01", "2023-05-07"],
        ...    name='special_events'
        ... )
        >>> tp.calendar_month(evset.node()).run(evset)
        indexes: ...
        features: [('calendar_month', int32)]
        events:
            (4 events):
                timestamps: [...]
                'calendar_month': [2 2 3 5]
        ...

        ```

    Args:
        sampling: Node with unix timestamp sampling.

    Returns:
        Single feature with the month each timestamp in `sampling` belongs to.
    """
    return CalendarMonthOperator(sampling).outputs["output"]
