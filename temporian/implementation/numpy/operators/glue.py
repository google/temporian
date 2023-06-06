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

"""Implementation for the Glue operator."""

from typing import Dict, List

from temporian.core.operators.glue import GlueOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet, IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation


class GlueNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the glue operator."""

    def __init__(self, operator: GlueOperator):
        super().__init__(operator)
        assert isinstance(operator, GlueOperator)

    def __call__(
        self,
        **inputs: EventSet,
    ) -> Dict[str, EventSet]:
        assert isinstance(self.operator, GlueOperator)
        output_schema = self.output_schema("output")

        # convert input evest dict to list of evsets
        evsets: List[EventSet] = list(
            list(zip(*sorted(list(inputs.items()))))[1]
        )
        if len(evsets) < 2:
            raise ValueError(
                f"Glue operator cannot be called on a {len(evsets)} event sets."
            )

        dst_evset = EventSet(data={}, schema=output_schema)
        for index_key, index_data in evsets[0].data.items():
            features = []
            for evset in evsets:
                features.extend(evset.data[index_key].features)

            dst_evset[index_key] = IndexData(
                timestamps=index_data.timestamps,
                features=features,
                schema=output_schema,
            )

        return {"output": dst_evset}


implementation_lib.register_operator_implementation(
    GlueOperator, GlueNumpyImplementation
)
