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

from temporian.core.data.duration import duration_abbreviation
from temporian.core.operators.lag import LagOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class LagNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: LagOperator) -> None:
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        duration = self.operator.attributes()["duration"]

        sampling_data = {}
        output_data = {}

        prefix = "lag" if duration > 0 else "leak"
        duration_str = duration_abbreviation(duration)

        for index, timestamps in event.sampling.data.items():
            sampling_data[index] = timestamps + duration
            output_data[index] = []
            for feature in event.data[index]:
                new_feature = NumpyFeature(
                    data=feature.data,
                    name=f"{prefix}[{duration_str}]_{feature.name}",
                )
                output_data[index].append(new_feature)

        new_sampling = NumpySampling(
            data=sampling_data,
            index=event.sampling.index.copy(),
        )
        output_event = NumpyEvent(data=output_data, sampling=new_sampling)

        return {"event": output_event}


implementation_lib.register_operator_implementation(
    LagOperator, LagNumpyImplementation
)
