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

from temporian.core.operators.lag import LagOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class LagNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: LagOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, LagOperator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, LagOperator)
        output_schema = self.output_schema("output")

        # gather operator attributes
        duration = self.operator.duration

        # create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        # fill output EventSet data
        for index_key, index_data in input.data.items():
            output_evset[index_key] = IndexData(
                index_data.features,
                index_data.timestamps + duration,
                schema=output_schema,
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    LagOperator, LagNumpyImplementation
)
