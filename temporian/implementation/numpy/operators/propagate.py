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

from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.core.operators.propagate import Propagate
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class PropagateNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the propagate operator."""

    def __init__(self, operator: Propagate) -> None:
        super().__init__(operator)
        assert isinstance(operator, Propagate)

    def __call__(
        self, event: NumpyEvent, sampling: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        dst_data = {}

        for sampling_index in sampling.data:
            # Compute the event index
            src_index = tuple(
                [sampling_index[i] for i in self.operator.index_mapping]
            )

            # Find the source data
            if src_index not in event.data:
                # TODO: Add option to skip non matched indexes.
                raise ValueError(f'Cannot find index "{src_index}" in "event".')

            dst_data[sampling_index] = event.data[src_index]

        output_event = NumpyEvent(
            data=dst_data,
            feature_names=event.feature_names,
            index_names=sampling.index_names,
            is_unix_timestamp=sampling.is_unix_timestamp,
        )
        return {"event": output_event}


implementation_lib.register_operator_implementation(
    Propagate, PropagateNumpyImplementation
)
