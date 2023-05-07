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

from typing import Dict

from temporian.core.operators.glue import GlueOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class GlueNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the glue operator."""

    def __init__(self, operator: GlueOperator):
        super().__init__(operator)
        assert isinstance(operator, GlueOperator)

    def __call__(
        self,
        **inputs: Dict[str, EventSet],
    ) -> Dict[str, EventSet]:
        """Glues a dictionary of EventSets.

        The output features are ordered by the argument name of the event.

        Example:
            If evset_dict is {"evset_0": X, "evset_3": Z, "evset_2": Y}
            output is guarenteed to be [X, Y, Z].
        """
        # convert input evest dict to list of evsets
        evsets = list(list(zip(*sorted(list(inputs.items()))))[1])
        if len(evsets) < 2:
            raise ValueError(
                f"Glue operator cannot be called on a {len(evsets)} event sets."
            )

        # create output event set
        dst_evset = EventSet(
            data={},
            feature_names=[
                feature_name
                for evset in evsets
                for feature_name in evset.feature_names
            ],
            index_names=evsets[0].index_names,
            is_unix_timestamp=evsets[0].is_unix_timestamp,
        )
        # fill output event set data
        for index_key, index_data in evsets[0].iterindex():
            dst_evset[index_key] = index_data
            for evset in evsets[1:]:
                dst_evset[index_key].features.extend(evset[index_key].features)

        return {"output": dst_evset}


implementation_lib.register_operator_implementation(
    GlueOperator, GlueNumpyImplementation
)
