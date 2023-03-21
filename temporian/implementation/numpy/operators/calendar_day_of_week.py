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

from typing import Any
from datetime import datetime

from temporian.core.operators.calendar_day_of_week import (
    CalendarDayOfWeekOperator,
)
from temporian.implementation.numpy.operators.calendar_base import (
    BaseCalendarNumpyImplementation,
)


class CalendarDayOfWeekNumpyImplementation(BaseCalendarNumpyImplementation):
    """Numpy implementation of the calendar_day_of_week operator."""

    def __init__(self, operator: CalendarDayOfWeekOperator) -> None:
        super().__init__(operator)

    def _get_value_from_datetime(self, dt: datetime) -> Any:
        return dt.weekday()

    @property
    def _output_feature_name(self) -> str:
        return "calendar_day_of_week"
