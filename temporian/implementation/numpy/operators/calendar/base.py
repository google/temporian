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
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseCalendarNumpyImplementation(OperatorImplementation):
    """Abstract base class to implement common logic of numpy implementation of
    calendar operators."""

    def __init__(self, operator: BaseCalendarOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, BaseCalendarOperator)

    def __call__(self, sampling: NumpyEvent) -> Dict[str, NumpyEvent]:
        # create destination event
        dst_event = NumpyEvent(
            data={},
            feature_names=[self.operator.output_feature_name],
            index_names=sampling.index_names,
            is_unix_timestamp=True,
        )
        for index_key, index_data in sampling.iterindex():
            value = np.array(
                [
                    self._get_value_from_datetime(
                        datetime.fromtimestamp(ts, tz=timezone.utc)
                    )
                    for ts in index_data.timestamps
                ]
            ).astype(
                np.int32
            )  # TODO: parametrize output dtype

            dst_event[index_key] = IndexData(value, index_data.timestamps)

        return {"event": dst_event}

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
                int32 by __call__.
        """
