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

"""Calendar year operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.operators.base import EventSetOrNode
from temporian.core.operators.calendar.base import BaseCalendarOperator


class CalendarYearOperator(BaseCalendarOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "CALENDAR_YEAR"

    @classmethod
    def output_feature_name(cls) -> str:
        return "calendar_year"


operator_lib.register_operator(CalendarYearOperator)


@compile
def calendar_year(sampling: EventSetOrNode) -> EventSetOrNode:
    """Obtains the year the timestamps in a node's sampling are in.

    Features in the input node are ignored, only the timestamps are used and
    they must be unix timestamps (`is_unix_timestamp=True`).

    Usage example:
        ```python
        >>> evset = tp.event_set(
        ...    timestamps=["2021-02-04", "2022-02-20", "2023-03-01", "2023-05-07"],
        ...    name='random_moments'
        ... )
        >>> tp.calendar_year(evset.node()).run(evset)
        indexes: ...
        features: [('calendar_year', int32)]
        events:
            (4 events):
                timestamps: [...]
                'calendar_year': [2021 2022 2023 2023]
        ...

        ```

    Args:
        sampling: Node to get the years from.

    Returns:
        Single feature with the year each timestamp in `sampling` belongs to.
    """
    return CalendarYearOperator(sampling).outputs["output"]
