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

from temporian.implementation.numpy.data.event import (
    NumpyEvent,
    NumpyFeature,
)
from temporian.core.operators.prefix import Prefix
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class PrefixNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the prefix operator."""

    def __init__(self, operator: Prefix) -> None:
        assert isinstance(operator, Prefix)
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        prefix = self._operator.prefix()
        dst_event = NumpyEvent(data={}, sampling=event.sampling)

        # For each index value
        for index, features in event.data.items():
            dst_event.data[index] = [
                NumpyFeature(prefix + feature.name, feature.data)
                for feature in features
            ]

        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    Prefix, PrefixNumpyImplementation
)
