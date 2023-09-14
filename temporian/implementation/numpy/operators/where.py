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


"""Implementation for the Where operator."""


from typing import Dict, Optional
import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.where import Where
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class WhereNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: Where) -> None:
        assert isinstance(operator, Where)
        super().__init__(operator)

    def __call__(
        self,
        input: EventSet,
        on_true: Optional[EventSet] = None,
        on_false: Optional[EventSet] = None,
    ) -> Dict[str, EventSet]:
        assert isinstance(self.operator, Where)

        # Create output EventSet
        output_schema = self.output_schema("output")
        output_evset = EventSet(data={}, schema=output_schema)

        # Single value sources, None if EventSets are provided.
        on_true_source = self.operator.on_true
        on_false_source = self.operator.on_false

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            # EventSet sources instead of single values
            if on_true is not None:
                on_true_source = on_true.data[index_key].features[0]
            if on_false is not None:
                on_false_source = on_false.data[index_key].features[0]

            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=[
                        np.where(
                            index_data.features[0],
                            on_true_source,
                            on_false_source,
                        )
                    ],
                    timestamps=index_data.timestamps,
                    schema=output_schema,
                ),
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    Where, WhereNumpyImplementation
)
