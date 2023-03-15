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

from typing import Dict
from datetime import datetime

import numpy as np
from temporian.core.operators.calendar_day import CalendarDayOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature


class CalendarDayNumpyImplementation:
    def __init__(self, operator: CalendarDayOperator) -> None:
        super().__init__()
        self.operator = operator

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        data = {}
        for index, timestamps in event.sampling.data.items():
            print(timestamps)
            print(timestamps.dtype)
            days = np.array(
                [datetime.fromtimestamp(ts).day for ts in timestamps]
            ).astype(np.int32)

            data[index] = [
                NumpyFeature(
                    data=days,
                    name="calendar_day",
                )
            ]

        return {"event": NumpyEvent(data=data, sampling=event.sampling)}
