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

"""Calendar day of year operator class and public API function definitions."""

from typing import Union
from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import EventSetNode
from temporian.core.operators.calendar.base import BaseCalendarOperator
from temporian.core.typing import EventSetOrNode


class CalendarDayOfYearOperator(BaseCalendarOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "CALENDAR_DAY_OF_YEAR"

    @classmethod
    def output_feature_name(cls) -> str:
        return "calendar_day_of_year"


operator_lib.register_operator(CalendarDayOfYearOperator)


@compile
def calendar_day_of_year(
    sampling: EventSetOrNode, tz: Union[str, float, int] = 0
) -> EventSetOrNode:
    assert isinstance(sampling, EventSetNode)
    utc_offset = BaseCalendarOperator.convert_to_utc_offset(tz)

    return CalendarDayOfYearOperator(sampling, utc_offset).outputs["output"]
