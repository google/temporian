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
from temporian.core.data.node import Node
from temporian.core.operators.calendar.base import BaseCalendarOperator


class CalendarDayOfMonthOperator(BaseCalendarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "CALENDAR_DAY_OF_MONTH"

    @classmethod
    @property
    def output_feature_name(cls) -> str:
        return "calendar_day_of_month"


operator_lib.register_operator(CalendarDayOfMonthOperator)


def calendar_day_of_month(sampling: Node) -> Node:
    """Obtains the day of month the timestamps in a node's sampling are in.

    Features in input node are ignored. Output feature contains numbers between
    1 and 31.

    Args:
        sampling: Node to get the days of month from.

    Returns:
        Node with a single feature corresponding to the day of the month each
        timestamp in `node`'s sampling belongs to, with the same sampling as
        `node`.
    """
    return CalendarDayOfMonthOperator(sampling).outputs["node"]
