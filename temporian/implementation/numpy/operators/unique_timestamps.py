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

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.core.operators.unique_timestamps import UniqueTimestamps
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.implementation.numpy.data.sampling import NumpySampling


class UniqueTimestampsNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: UniqueTimestamps) -> None:
        assert isinstance(operator, UniqueTimestamps)
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        sampling_data = {}
        output_data = {}

        for index, timestamps in event.sampling.data.items():
            # TODO: Optimize using the fact that "timestamps" is sorted.
            sampling_data[index] = np.unique(timestamps)
            output_data[index] = []

        new_sampling = NumpySampling(
            data=sampling_data,
            index=event.sampling.index.copy(),
            is_unix_timestamp=event.sampling.is_unix_timestamp,
        )
        output_event = NumpyEvent(data=output_data, sampling=new_sampling)

        return {"event": output_event}


implementation_lib.register_operator_implementation(
    UniqueTimestamps, UniqueTimestampsNumpyImplementation
)
