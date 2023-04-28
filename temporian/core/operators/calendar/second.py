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

"""Calendar second operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.calendar.base import BaseCalendarOperator


class CalendarSecondOperator(BaseCalendarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "CALENDAR_SECOND"

    @classmethod
    @property
    def output_feature_name(cls) -> str:
        return "calendar_second"


operator_lib.register_operator(CalendarSecondOperator)


def calendar_second(sampling: Event) -> Event:
    """Obtains the second the timestamps in an event's sampling are in.

    Features in input event are ignored. Output feature contains numbers between
    0 and 59.

    Args:
        sampling: Event to get the seconds from.

    Returns:
        Event with a single feature corresponding to the second each timestamp
        in `event`'s sampling belongs to, with the same sampling as `event`.
    """
    return CalendarSecondOperator(sampling).outputs["event"]
