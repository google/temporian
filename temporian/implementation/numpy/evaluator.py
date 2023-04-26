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

import sys
import time

from typing import Dict, List

from temporian.core.data import event
from temporian.core.operators import base
from temporian.implementation.numpy.data import event as numpy_event
from temporian.implementation.numpy import implementation_lib


def evaluate_schedule(
    inputs: Dict[event.Event, numpy_event.NumpyEvent],
    schedule: List[base.Operator],
    verbose: int,
    check_execution: bool,
) -> Dict[event.Event, numpy_event.NumpyEvent]:
    """Evaluates a schedule on a dictionary of input event data.

    Args:
        inputs: Mapping of core Events to materialized NumpyEvents.
        schedule: Sequence of operators to apply on the data.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.
        check_execution: If `True`, data of the intermediate results of the
            operators is checked against its expected structure and raises if
            it differs.
    """
    data = {**inputs}

    for operator_idx, operator in enumerate(schedule):
        operator_def = operator.definition()

        # Get implementation
        implementation_cls = implementation_lib.get_implementation_class(
            operator_def.key
        )

        # Instantiate implementation
        implementation = implementation_cls(operator)

        if verbose == 1:
            print(
                (
                    f"    {operator_idx+1} / {len(schedule)}:"
                    f" {operator.operator_key()}"
                ),
                file=sys.stderr,
                end="",
            )
        elif verbose >= 2:
            print(
                f"Run {operator}",
                file=sys.stderr,
            )

        # Construct operator inputs
        operator_inputs = {
            input_key: data[input_event]
            for input_key, input_event in operator.inputs().items()
        }

        # Compute output
        begin_time = time.perf_counter()
        if check_execution:
            operator_outputs = implementation.call(**operator_inputs)
        else:
            operator_outputs = implementation(**operator_inputs)
        end_time = time.perf_counter()

        if verbose == 1:
            print(f" [{end_time - begin_time:.5f} s]", file=sys.stderr)
        elif verbose >= 2:
            print(f"Duration: {end_time - begin_time} s", file=sys.stderr)

        # materialize data in output events
        for output_key, output_event in operator.outputs().items():
            data[output_event] = operator_outputs[output_key]

    # TODO: Only return the required data.
    # TODO: Un-allocate not used anymore object.
    return data
