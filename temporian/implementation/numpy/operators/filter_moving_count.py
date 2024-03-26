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


"""Implementation for the FilterMaxMovingCount operator."""


from typing import Dict
import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.filter_moving_count import (
    FilterMaxMovingCount,
)
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.implementation.numpy_cc.operators import operators_cc


class FilterMaxMovingCountNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: FilterMaxMovingCount) -> None:
        assert isinstance(operator, FilterMaxMovingCount)
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, FilterMaxMovingCount)

        output_schema = self.output_schema("output")

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        window_length = self.operator.window_length

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            sampling_idx = operators_cc.filter_moving_count(
                index_data.timestamps,
                window_length=window_length,
            )

            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=[f[sampling_idx] for f in index_data.features],
                    timestamps=index_data.timestamps[sampling_idx],
                    schema=output_schema,
                ),
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    FilterMaxMovingCount, FilterMaxMovingCountNumpyImplementation
)
