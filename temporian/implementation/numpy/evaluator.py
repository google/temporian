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

from typing import Dict, List

from temporian.core.data import event
from temporian.core.operators import base
from temporian.implementation.numpy.data import event as numpy_event
from temporian.implementation.numpy.operators import core_mapping


def evaluate_schedule(
    data: Dict[event.Event, numpy_event.NumpyEvent],
    schedule: List[base.Operator],
) -> Dict[event.Event, numpy_event.NumpyEvent]:
    for operator in schedule:
        operator_def = operator.definition()
        # get implementation
        implementation = core_mapping.OPERATOR_IMPLEMENTATIONS[
            operator_def.key
        ](operator)

        # construct operator inputs
        operator_inputs = {
            input_key: data[input_event]
            for input_key, input_event in operator.inputs().items()
        }

        # compute output
        operator_outputs = implementation(**operator_inputs)

        # materialize data in output events
        for output_key, output_event in operator.outputs().items():
            data[output_event] = operator_outputs[output_key]

    return data
