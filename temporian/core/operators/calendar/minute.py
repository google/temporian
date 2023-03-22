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

"""Calendar minute operator."""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.calendar.base import BaseCalendarOperator


class CalendarMinuteOperator(BaseCalendarOperator):
    """
    Calendar operator to obtain the minute each timestamp belongs to.
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "CALENDAR_MINUTE"

    @classmethod
    @property
    def output_feature_name(cls) -> str:
        return "calendar_minute"


operator_lib.register_operator(CalendarMinuteOperator)


def calendar_minute(sampling: Event) -> Event:
    """Obtain the minute each of the timestamps in an event's sampling belongs
    to. Features in input event are ignored. Output feature contains numbers
    between 0 and 59.

    Args:
        sampling: the event to get the minutes from.

    Returns:
        event with a single feature corresponding to the minute each timestamp
            in `event`'s sampling belongs to, with the same sampling as `event`.
    """
    return CalendarMinuteOperator(sampling).outputs()["event"]
