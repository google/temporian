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


"""Implementation for the Unique operator."""


from typing import Dict
import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.unique import Unique
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation

class UniqueNumpyImplementation(OperatorImplementation):

    def __init__(self, operator: Unique) -> None:
        assert isinstance(operator, Unique)
        super().__init__(operator)

    def __call__(
        self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, Unique)

        output_schema = self.output_schema("output")

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            if not index_data.features:
                # Handle empty features case
                output_evset.set_index_value(
                    index_key,
                    IndexData(
                        features=[],
                        timestamps=index_data.timestamps,
                        schema=output_schema,
                    )
                )
                continue
                
            # As per issue #345: For this first version, only works on single-feature EventSets.
            feature_data = index_data.features[0]
            
            # np.unique with return_index=True returns the first occurrence index for each unique value.
            _, unique_indices = np.unique(feature_data, return_index=True)
            
            # Sort the indices to maintain the strict chronological order of timestamps
            unique_indices.sort()
            
            # Filter timestamps and features using the unique indices
            filtered_timestamps = index_data.timestamps[unique_indices]
            filtered_feature = feature_data[unique_indices]
            
            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=[filtered_feature],
                    timestamps=filtered_timestamps,
                    schema=output_schema,
                )
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    Unique, UniqueNumpyImplementation
)
