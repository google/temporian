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


"""Implementation for the Tick operator."""


from typing import Dict

import math
import numpy as np
from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.tick import Tick
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class TickNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: Tick) -> None:
        assert isinstance(operator, Tick)
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, Tick)

        output_schema = self.output_schema("output")

        # create output event set
        output_evset = EventSet(data={}, schema=output_schema)

        # fill output event set data
        for index_key, index_data in input.data.items():
            if len(index_data.timestamps) == 0:
                dst_timestamps = np.array([], dtype=np.float64)
            else:
                begin = index_data.timestamps[0]
                end = index_data.timestamps[-1]

                if self.operator.align:
                    save_begin = begin
                    begin = (
                        float(begin // self.operator.interval)
                        * self.operator.interval
                    )

                    if save_begin != begin:
                        begin += self.operator.interval

                dst_timestamps = np.arange(
                    begin,
                    np.nextafter(end, math.inf),
                    self.operator.interval,
                    dtype=np.float64,
                )

            output_evset[index_key] = IndexData(
                [],
                dst_timestamps,
                schema=output_schema,
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    Tick, TickNumpyImplementation
)
