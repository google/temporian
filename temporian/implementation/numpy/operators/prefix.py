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

from temporian.core.operators.prefix import Prefix
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class PrefixNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the prefix operator."""

    def __init__(self, operator: Prefix) -> None:
        super().__init__(operator)
        assert isinstance(operator, Prefix)

    def __call__(self, node: EventSet) -> Dict[str, EventSet]:
        # gather operator attributes
        prefix = self._operator.prefix()

        # create output evset
        dst_evset = EventSet(
            data=node.data,
            feature_names=[
                f"{prefix}{feature_name}" for feature_name in node.feature_names
            ],
            index_names=node.index_names,
            is_unix_timestamp=node.is_unix_timestamp,
        )
        return {"node": dst_evset}


implementation_lib.register_operator_implementation(
    Prefix, PrefixNumpyImplementation
)
