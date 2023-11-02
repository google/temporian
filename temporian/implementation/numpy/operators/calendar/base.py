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

from abc import abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict

import numpy as np

from temporian.core.operators.calendar.base import BaseCalendarOperator
from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseCalendarNumpyImplementation(OperatorImplementation):
    """Interface definition and common logic for numpy implementation of
    calendar operators."""

    def __init__(self, operator: BaseCalendarOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, BaseCalendarOperator)

    def __call__(self, sampling: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, BaseCalendarOperator)
        output_schema = self.output_schema("output")
        tzinfo = timezone(timedelta(hours=self.operator.utc_offset))

        # create destination EventSet
        dst_evset = EventSet(data={}, schema=output_schema)
        for index_key, index_data in sampling.data.items():
            value = np.array(
                [
                    self._get_value_from_datetime(
                        datetime.fromtimestamp(ts, tz=tzinfo)
                    )
                    for ts in index_data.timestamps
                ],
                dtype=np.int32,
            )

            dst_evset.set_index_value(
                index_key,
                IndexData([value], index_data.timestamps, schema=output_schema),
                normalize=False,
            )

        return {"output": dst_evset}

    @abstractmethod
    def _get_value_from_datetime(self, dt: datetime) -> int:
        """Gets the value of the datetime object that corresponds to each
        specific calendar operator.

        For example, calendar_day_of_month will return the datetime's day, and
        calendar_hour its hour.

        Returned value is converted to int32 by __call__.

        Args:
            dt: Datetime to get the value from.

        Returns:
            Numeric value for the datetime.
        """
