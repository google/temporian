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


"""Implementation for the UntilNext operator."""


from typing import Dict
import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.until_next import UntilNext
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.implementation.numpy_cc.operators import operators_cc


class UntilNextNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: UntilNext) -> None:
        assert isinstance(operator, UntilNext)
        super().__init__(operator)

    def __call__(
        self, input: EventSet, sampling: EventSet
    ) -> Dict[str, EventSet]:
        assert isinstance(self.operator, UntilNext)

        output_schema = self.output_schema("output")

        timeout = self.operator.timeout

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        empty_timestamps = np.array([], dtype=np.float64)

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            if index_key in sampling.data:
                sampling_timestamps = sampling.data[index_key].timestamps
            else:
                sampling_timestamps = empty_timestamps

            until_next_timestamps, until_next_values = operators_cc.until_next(
                index_data.timestamps, sampling_timestamps, timeout
            )
            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=[until_next_values],
                    timestamps=until_next_timestamps,
                    schema=output_schema,
                ),
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    UntilNext, UntilNextNumpyImplementation
)
