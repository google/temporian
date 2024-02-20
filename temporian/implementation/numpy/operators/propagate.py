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

import numpy as np

from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core.operators.propagate import Propagate
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.implementation.numpy.data import event_set
from temporian.implementation.numpy.data import dtype_normalization


class PropagateNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the propagate operator."""

    def __init__(self, operator: Propagate) -> None:
        super().__init__(operator)
        assert isinstance(operator, Propagate)

    def __call__(
        self, input: EventSet, sampling: EventSet
    ) -> Dict[str, EventSet]:
        assert isinstance(self.operator, Propagate)
        output_schema = self.output_schema("output")

        dst_data = {}

        for sampling_index in sampling.data:
            # Compute the EventSet's index
            src_index = tuple(
                [sampling_index[i] for i in self.operator.index_mapping]
            )

            # Find the source data
            if src_index not in input.data:
                # An empty index value.
                dst_data[sampling_index] = event_set.IndexData(
                    features=[
                        np.zeros(
                            shape=0,
                            dtype=dtype_normalization.tp_dtype_to_np_dtype(
                                feature.dtype
                            ),
                        )
                        for feature in output_schema.features
                    ],
                    timestamps=np.array([], dtype=np.float64),
                    schema=output_schema,
                )
            else:
                dst_data[sampling_index] = input.data[src_index]

        output_evset = EventSet(data=dst_data, schema=output_schema)
        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    Propagate, PropagateNumpyImplementation
)
