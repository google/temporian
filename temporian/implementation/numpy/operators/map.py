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


"""Implementation for the Map operator."""


from typing import Dict
import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.map import Map
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class MapNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: Map) -> None:
        assert isinstance(operator, Map)
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, Map)

        output_schema = self.output_schema("output")
        func = self.operator.func

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            # Iterate over features and apply func
            features = []
            for orig_feature in index_data.features:
                feature = np.empty_like(orig_feature)
                for i, value in enumerate(orig_feature):
                    feature[i] = func(value)
                features.append(feature)

            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=features,
                    timestamps=index_data.timestamps,
                    schema=output_schema,
                ),
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(Map, MapNumpyImplementation)
