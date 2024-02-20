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

import time
import sys
import time
import gc

from typing import Dict, Optional

from temporian.core.data.node import EventSetNode
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core.schedule import Schedule

# Loads all the numpy operator implementations
from temporian.implementation.numpy import operators as _impls


def run_schedule(
    inputs: Dict[EventSetNode, EventSet],
    schedule: Schedule,
    verbose: int,
    check_execution: bool,
    force_garbage_collector_interval: Optional[float] = 10,
) -> Dict[EventSetNode, EventSet]:
    """Evaluates a schedule on a dictionary of input
    [`EventSets`][temporian.EventSet].

    Args:
        inputs: Mapping of EventSetNodes to materialized EventSets.
        schedule: Sequence of operators to apply on the data.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.
        check_execution: If `True`, data of the intermediate results of the
            operators is checked against its expected structure and raises if
            it differs.
        force_garbage_collector_interval: If set, triggers the garbage
            collection every "force_garbage_collector_interval" seconds.
    """
    data = {**inputs}

    gc_being_time = time.time()

    num_steps = len(schedule.steps)
    for step_idx, step in enumerate(schedule.steps):
        operator_def = step.op.definition

        # Get implementation
        implementation_cls = implementation_lib.get_implementation_class(
            operator_def.key
        )

        # Instantiate implementation
        implementation = implementation_cls(step.op)

        if verbose == 1:
            print(
                f"    {step_idx+1} / {num_steps}: {step.op.operator_key()}",
                file=sys.stderr,
                end="",
                flush=True,
            )
        elif verbose >= 2:
            print("=============================", file=sys.stderr)
            print(
                f"{step_idx+1} / {num_steps}: Run {step.op}",
                file=sys.stderr,
                flush=True,
            )

        # Construct operator inputs
        operator_inputs = {
            input_key: data[input_node]
            for input_key, input_node in step.op.inputs.items()
        }

        if verbose >= 2:
            print(
                f"Inputs:\n{operator_inputs}\n",
                file=sys.stderr,
                flush=True,
            )

        # Compute output
        begin_time = time.perf_counter()
        if check_execution:
            operator_outputs = implementation.call(**operator_inputs)
        else:
            operator_outputs = implementation(**operator_inputs)
        end_time = time.perf_counter()

        if verbose == 1:
            print(
                f" [{end_time - begin_time:.5f} s]",
                file=sys.stderr,
                flush=True,
            )
        elif verbose >= 2:
            print(f"Outputs:\n{operator_outputs}\n", file=sys.stderr)
            print(
                f"Duration: {end_time - begin_time} s",
                file=sys.stderr,
                flush=True,
            )

        # materialize data in output nodes
        for output_key, output_node in step.op.outputs.items():
            output_evset = operator_outputs[output_key]
            output_evset._internal_node = output_node
            data[output_node] = output_evset

        # Release unused memory
        for node in step.released_nodes:
            assert node in data
            del data[node]

        if (
            force_garbage_collector_interval is not None
            and (time.time() - gc_being_time)
            >= force_garbage_collector_interval
        ):
            begin_gc = time.time()
            if verbose >= 1:
                print("Garbage collection", file=sys.stderr, flush=True, end="")
            gc.collect()
            gc_being_time = time.time()
            if verbose >= 1:
                print(
                    f" [{end_time - begin_gc:.5f} s]",
                    file=sys.stderr,
                    flush=True,
                )

    return data
