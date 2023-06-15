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

from typing import Dict

from temporian.core.data.node import Node
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core.schedule import Schedule

# Loads all the numpy operator implementations
from temporian.implementation.numpy import operators as _impls


def evaluate_schedule(
    inputs: Dict[Node, EventSet],
    schedule: Schedule,
    verbose: int,
    check_execution: bool,
) -> Dict[Node, EventSet]:
    """Evaluates a schedule on a dictionary of input event sets.

    Args:
        inputs: Mapping of nodes to materialized EventSets.
        schedule: Sequence of operators to apply on the data.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.
        check_execution: If `True`, data of the intermediate results of the
            operators is checked against its expected structure and raises if
            it differs.
    """
    data = {**inputs}

    num_operators = len(schedule.ordered_operators)
    for operator_idx, operator in enumerate(schedule.ordered_operators):
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
                    f"    {operator_idx+1} / {num_operators}:"
                    f" {operator.operator_key()}"
                ),
                file=sys.stderr,
                end="",
            )
        elif verbose >= 2:
            print("=============================", file=sys.stderr)
            print(
                f"{operator_idx+1} / {num_operators}: Run {operator}",
                file=sys.stderr,
            )

        # Construct operator inputs
        operator_inputs = {
            input_key: data[input_node]
            for input_key, input_node in operator.inputs.items()
        }

        if verbose >= 2:
            print(f"Inputs:\n{operator_inputs}\n", file=sys.stderr)

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
            print(f"Outputs:\n{operator_outputs}\n", file=sys.stderr)
            print(f"Duration: {end_time - begin_time} s", file=sys.stderr)

        # materialize data in output nodes
        for output_key, output_node in operator.outputs.items():
            data[output_node] = operator_outputs[output_key]

    # TODO: Only return the required data.
    # TODO: Un-allocate not used anymore object.
    return data
