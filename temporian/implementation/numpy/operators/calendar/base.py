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
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
from temporian.core.operators.calendar.base import BaseCalendarOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseCalendarNumpyImplementation(OperatorImplementation):
    """Interface definition and common logic for numpy implementation of
    calendar operators."""

    def __init__(self, operator: BaseCalendarOperator) -> None:
        super().__init__(operator)

    def __call__(self, sampling: NumpyEvent) -> Dict[str, NumpyEvent]:
        data = {}
        for index, timestamps in sampling.sampling.data.items():
            days = np.array(
                [
                    self._get_value_from_datetime(
                        datetime.fromtimestamp(ts, tz=timezone.utc)
                    )
                    for ts in timestamps
                ]
            ).astype(np.int32)

            data[index] = [
                NumpyFeature(
                    data=days,
                    name=self.operator.output_feature_name,
                )
            ]

        return {"event": NumpyEvent(data=data, sampling=sampling.sampling)}

    @abstractmethod
    def _get_value_from_datetime(self, dt: datetime) -> Any:
        """Gets the value of the datetime object that corresponds to each
        specific calendar operator.

        For example, calendar_day_of_month will take the datetime's day, and
        calendar_hour will take the its hour.

        Returned value is be converted to int32 by __call__.

        Args:
            dt: the datetime to get the value from.

        Returns:
            The numeric value for the datetime.
        """
