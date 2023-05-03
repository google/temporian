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
from temporian.core.data.node import Node
from temporian.core.operators.calendar.base import BaseCalendarOperator


class CalendarDayOfWeekOperator(BaseCalendarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "CALENDAR_DAY_OF_WEEK"

    @classmethod
    @property
    def output_feature_name(cls) -> str:
        return "calendar_day_of_week"


operator_lib.register_operator(CalendarDayOfWeekOperator)


def calendar_day_of_week(sampling: Node) -> Node:
    """Obtains the day of the week the timestamps in a node's sampling are in.

    Features in input node are ignored. Output feature contains numbers from 0
    (Monday) to 6 (Sunday).

    Args:
        sampling: Node to get the days of week from.

    Returns:
        Node with a single feature corresponding to the day of the week each
        timestamp in `sampling`'s sampling belongs to, with the same sampling as
        `sampling`.
    """
    return CalendarDayOfWeekOperator(sampling).outputs["output"]
