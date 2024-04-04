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


"""Implementation for the FilterEmptyIndex operator."""


from typing import Dict

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.filter_empty_index import FilterEmptyIndex
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class FilterEmptyIndexNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: FilterEmptyIndex) -> None:
        assert isinstance(operator, FilterEmptyIndex)
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, FilterEmptyIndex)

        output_schema = self.output_schema("output")

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            if len(index_data.timestamps) > 0:
                output_evset.set_index_value(
                    index_key,
                    IndexData(
                        features=index_data.features,
                        timestamps=index_data.timestamps,
                        schema=output_schema,
                    ),
                )
        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    FilterEmptyIndex, FilterEmptyIndexNumpyImplementation
)
