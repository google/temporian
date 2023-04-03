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

"""Evaluator module."""

import time
import sys
import pathlib
from typing import Any, Dict, List, Set, Union
from collections import defaultdict

from temporian.core.data.event import Event
from temporian.core.operators import base
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.core import processor as processor_lib
from temporian.implementation.numpy import evaluator as numpy_evaluator

AvailableBackends = Any
Data = Dict[Event, Union[str, pathlib.Path, NumpyEvent]]
Query = Union[Event, List[Event], Dict[str, Event]]
Result = Union[NumpyEvent, List[NumpyEvent], Dict[str, NumpyEvent]]


def evaluate(
    query: Query,
    input_data: Data,
    verbose: int = 1,
    check_execution: bool = True,
) -> Result:
    """Evaluates a query on data.

    Args:
        query: Events to compute. Support event, dict and list of events.
        input_data: Dictionary of event and event values to use for the
          computation.
        verbose: If >0, prints details about the execution on the standard error
          output. The larger the number, the more information is displayed.
        check_execution: If true, the input and output of the op implementation
          are validated to check any bug in the library internal code. If false,
          checks are skipped.

    TODO: Create an internal configuration object for options such as
    "check_execution".

    Returns:
        An object with the same structure as "event" containing the results. For
        instance, if "event" is a dictionary of events, the returned object
        will be a dictionary of event results. If "event" is a list of events,
        the returned value will be a list of event values with the same order.
    """

    begin_time = time.perf_counter()

    # Normalize the user query into a list query events.
    normalized_query: List[Event] = {}

    if isinstance(query, Event):
        # The query is a single value
        normalized_query = [query]

    elif isinstance(query, list):
        # The query is a list
        normalized_query = query

    elif isinstance(query, dict):
        # The query is a dictionary
        normalized_query = list(query.values())

    else:
        # TODO: improve error message
        raise TypeError(
            f"schedule_graph query argument must be one of {Query}. Received"
            f" {type(query)}."
        )

    if verbose >= 1:
        print("Build schedule", file=sys.stderr)

    # Schedule execution
    input_events = list(input_data.keys())
    schedule = build_schedule(
        inputs=input_events, outputs=normalized_query, verbose=verbose
    )

    if verbose == 1:
        print(
            f"Run {len(schedule)} operators",
            file=sys.stderr,
        )

    elif verbose >= 2:
        print("Schedule:\n", schedule, file=sys.stderr)

    # Evaluate schedule
    #
    # Note: "outputs" is a dictionary of event (including the query events) to
    # event data.
    outputs = numpy_evaluator.evaluate_schedule(
        input_data, schedule, verbose=verbose, check_execution=check_execution
    )

    end_time = time.perf_counter()

    if verbose == 1:
        print(f"Execution in {end_time - begin_time:.5f} s", file=sys.stderr)

    # Convert the result "outputs" into the same format as the query.
    if isinstance(query, Event):
        return outputs[query]

    elif isinstance(query, list):
        return [outputs[k] for k in query]

    elif isinstance(query, dict):
        return {
            query_key: outputs[query_evt]
            for query_key, query_evt in query.items()
        }

    else:
        raise RuntimeError("Unexpected case")


def build_schedule(
    inputs: List[Event],
    outputs: List[Event],
    verbose: int = 0,
) -> List[base.Operator]:
    """Calculates which operators need to be executed in which order to
    compute a set of output events given a set of input events.

    This implementation is based on Kahn's algorithm.

    Args:
        inputs: Input events.
        outputs: Output events.
        verbose: If >0, prints details about the execution on the standard error
          output. The larger the number, the more information is displayed.

    Returns:
        Ordered list of operators, such that the first operator should be
        computed before the second, second before the third, etc.
    """

    def list_to_dict(l: List[Any]) -> Dict[str, Any]:
        """Converts a list into a dict with a text index key."""
        return {str(i): x for i, x in enumerate(l)}

    # List all events and operators in between inputs and outputs.
    #
    # Fails if the outputs cannot be computed from the inputs e.g. some inputs
    # are missing.
    processor = processor_lib.infer_processor(
        list_to_dict(inputs), list_to_dict(outputs)
    )

    if verbose >= 2:
        print("Processor:\n", processor, file=sys.stderr)

    # Sequence of operators to execute. This is the result of the
    # "build_schedule" function.
    planned_ops: List[base.Operator] = []

    # Operators ready to be computed (i.e. ready to be added to "planned_ops")
    # as all their inputs are already computed by "planned_ops" or specified by
    # "inputs".
    ready_ops: Set[base.Operator] = set()

    # "event_to_op[e]" is the list of operators with event "e" as input.
    event_to_op: Dict[Event, List[base.Operator]] = defaultdict(lambda: [])

    # "op_to_num_pending_inputs[op]" is the number of "not yet scheduled" inputs
    # of operator "op". Operators in "op_to_num_pending_inputs" have not yet
    # scheduled.
    op_to_num_pending_inputs: Dict[base.Operator, int] = defaultdict(lambda: 0)

    # Compute "event_to_op" and "op_to_num_pending_inputs".
    inputs_set = set(inputs)
    for op in processor.operators():
        num_pending_inputs = 0
        for input_event in op.inputs().values():
            if input_event in inputs_set:
                # This input is already available
                continue
            event_to_op[input_event].append(op)
            num_pending_inputs += 1
        if num_pending_inputs == 0:
            # Ready to be scheduled
            ready_ops.add(op)
        else:
            # Some of the inputs are missing.
            op_to_num_pending_inputs[op] = num_pending_inputs

    # Compute the schedule
    while ready_ops:
        # Get an op ready to be scheduled
        op = ready_ops.pop()

        # Schedule the op
        planned_ops.append(op)

        # Update all the ops that depends on "op". Enlist the ones that are
        # ready to be computed
        for output in op.outputs().values():
            if output not in event_to_op:
                continue
            for new_op in event_to_op[output]:
                # "new_op" depends on the result of "op".
                assert new_op in op_to_num_pending_inputs
                num_missing_inputs = op_to_num_pending_inputs[new_op] - 1
                op_to_num_pending_inputs[new_op] = num_missing_inputs
                assert num_missing_inputs >= 0

                if num_missing_inputs == 0:
                    # "new_op" can be computed
                    ready_ops.add(new_op)
                    del op_to_num_pending_inputs[new_op]

    assert not op_to_num_pending_inputs

    return planned_ops
