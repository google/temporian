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


"""Implementation for the SelectIndexValues operator."""


from typing import Dict
from temporian.implementation.numpy.data.dtype_normalization import (
    normalize_index_key,
)

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.select_index_values import SelectIndexValues
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class SelectIndexValuesNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: SelectIndexValues) -> None:
        assert isinstance(operator, SelectIndexValues)
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, SelectIndexValues)

        output_schema = self.output_schema("output")

        keys = self.operator.keys

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        # TODO: remove this behavior when adding number/fraction params
        # and force one of the params to not be None
        if keys is None:
            keys = []

        # Fill output EventSet's data
        for key in keys:
            key = normalize_index_key(key)
            index_data = input.data[key]

            output_evset.set_index_value(
                key,
                IndexData(
                    features=index_data.features,
                    timestamps=index_data.timestamps,
                    schema=output_schema,
                ),
                normalize=False,
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    SelectIndexValues, SelectIndexValuesNumpyImplementation
)
