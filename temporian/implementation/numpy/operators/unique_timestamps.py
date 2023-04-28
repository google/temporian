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

"""Implementation for the UniqueTimestamps operator."""

from typing import Dict

import numpy as np

from temporian.core.operators.unique_timestamps import UniqueTimestamps
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class UniqueTimestampsNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: UniqueTimestamps) -> None:
        super().__init__(operator)
        assert isinstance(operator, UniqueTimestamps)

    def __call__(self, evset: EventSet) -> Dict[str, EventSet]:
        evset = EventSet(
            data={
                index_key: IndexData([], np.unique(index_data.timestamps))
                for index_key, index_data in evset.iterindex()
            },
            feature_names=[],
            index_names=evset.index_names,
            is_unix_timestamp=evset.is_unix_timestamp,
        )
        return {"node": evset}


implementation_lib.register_operator_implementation(
    UniqueTimestamps, UniqueTimestampsNumpyImplementation
)
