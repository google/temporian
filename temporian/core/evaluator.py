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

import pathlib
from typing import Any, Dict, List, Set, Union
from collections import defaultdict

from temporian.core import backends
from temporian.core.data.event import Event
from temporian.core.operators import base
from temporian.implementation.numpy.data import event as numpy_event
from temporian.core import processor as processor_lib

AvailableBackends = Any
Data = Dict[Event, Union[str, pathlib.Path, numpy_event.NumpyEvent]]
Query = Union[Event, List[Event], Set[Event]]


def evaluate(
    query: Query,
    input_data: Data,
    backend: AvailableBackends = "numpy",
) -> Dict[Event, Any]:
    """Evaluates a query on data."""

    # Normalize query
    normalized_query: List[Event] = {}
    if isinstance(query, Event):
        normalized_query = [query]

    elif isinstance(query, set):
        normalized_query = list(query)

    elif isinstance(query, list):
        normalized_query = query

    else:
        # TODO: improve error message
        raise TypeError(
            f"schedule_graph query argument must be one of {Query}. Received"
            f" {type(query)}."
        )

    # Select backend
    selected_backend = backends.BACKENDS[backend]
    event = selected_backend["event"]
    evaluate_schedule_fn = selected_backend["evaluate_schedule_fn"]
    read_csv_fn = selected_backend["read_csv_fn"]

    # input data is a list of events, create a dictionary with the event and the event data
    input_data = {event: event.data for event in input_data}

    # Schedule execution
    input_events = list(input_data.keys())
    schedule = build_schedule(inputs=input_events, outputs=normalized_query)

    # materialize input data. TODO: separate this logic
    materialized_input_data = {
        input_event: (
            input_event_spec
            if isinstance(input_event_spec, event)
            else read_csv_fn(input_event_spec, input_event.sampling())
        )
        for input_event, input_event_spec in input_data.items()
    }
    # evaluate schedule
    outputs = evaluate_schedule_fn(materialized_input_data, schedule)

    return {event: outputs[event] for event in normalized_query}


def build_schedule(
    inputs: List[Event], outputs: List[Event]
) -> List[base.Operator]:
    """Calculates which operators need to be executed in which order to
    compute a set of output events given a set of input events.

    This implementation is based on Kahn's algorithm.

    Args:
        inputs: Input events.
        outputs: Output events.

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
