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
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class SelectNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the select operator."""

    def __init__(self, operator: SelectOperator) -> None:
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        feature_names = self.operator.attributes()["feature_names"]

        output_event = NumpyEvent(
            {
                index_value: [
                    feature
                    for feature in features
                    if feature.name in feature_names
                ]
                for index_value, features in event.data.items()
            },
            event.sampling,
        )
        return {"event": output_event}


implementation_lib.register_operator_implementation(
    SelectOperator, SelectNumpyImplementation
)
