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

from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime, timezone

import numpy as np
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature


class BaseCalendarNumpyImplementation(ABC):
    """Abstract base class to implement common logic of numpy implementation of
    calendar operators."""

    def __call__(self, sampling: NumpyEvent) -> Dict[str, NumpyEvent]:
        data = {}
        for index, timestamps in sampling.sampling.data.items():
            days = np.array(
                [
                    datetime.fromtimestamp(ts, tz=timezone.utc).day
                    for ts in timestamps
                ]
            ).astype(np.int32)

            data[index] = [
                NumpyFeature(
                    data=days,
                    name="calendar_day_of_month",
                )
            ]

        return {"event": NumpyEvent(data=data, sampling=sampling.sampling)}

    @abstractmethod
    def _get_value_from_datetime(self, dt: datetime) -> Any:
        """Get the value of the datetime object that corresponds to each
        specific calendar operator. E.g., calendar_day_of_month will take the
        datetime's day, and calendar_hour will take the its hour.

        Must be implemented by subclasses.

        Args:
            dt: the datetime to get the value from.

        Returns:
            Any: the numeric value for the datetime. Will be converted to
                float32 by __call__.
        """
