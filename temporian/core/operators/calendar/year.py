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
from temporian.core.data.node import Node
from temporian.core.operators.calendar.base import BaseCalendarOperator


class CalendarYearOperator(BaseCalendarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "CALENDAR_YEAR"

    @classmethod
    @property
    def output_feature_name(cls) -> str:
        return "calendar_year"


operator_lib.register_operator(CalendarYearOperator)


def calendar_year(sampling: Node) -> Node:
    """Obtains the year the timestamps in a node's sampling are in.

    Features in input node are ignored.

    Args:
        sampling: Node to get the years from.

    Returns:
        Node with a single feature corresponding to the year each timestamp in
        `sampling`'s sampling belongs to, with the same sampling as `sampling`.
    """
    return CalendarYearOperator(sampling).outputs["output"]
