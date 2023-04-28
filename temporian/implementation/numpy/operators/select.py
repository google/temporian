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

from temporian.core.operators.select import SelectOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class SelectNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the select operator."""

    def __init__(self, operator: SelectOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, SelectOperator)

    def __call__(self, evset: EventSet) -> Dict[str, EventSet]:
        # gather operator attributes
        feature_names = self._operator.feature_names

        # get feature indexes to be selected
        feature_idxs = [
            evset.feature_names.index(feature_name)
            for feature_name in feature_names
        ]
        # create output event set
        output_evset = EventSet(
            data={},
            feature_names=feature_names,
            index_names=evset.index_names,
            is_unix_timestamp=evset.is_unix_timestamp,
        )
        # select feature index key-wise
        for index_key, index_data in evset.iterindex():
            output_evset[index_key] = IndexData(
                [index_data.features[idx] for idx in feature_idxs],
                index_data.timestamps,
            )

        return {"node": output_evset}


implementation_lib.register_operator_implementation(
    SelectOperator, SelectNumpyImplementation
)
