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

from temporian.core import backends
from temporian.core.data.event import Event
from temporian.core.operators import base
from temporian.implementation.pandas.data import event as pandas_event

# TODO: Use typing.Literal[tuple(backends.BACKENDS.keys())]
AvailableBackends = Any
Data = Dict[Event, Union[str, pathlib.Path, pandas_event.PandasEvent]]
Query = Union[Event, List[Event]]


def evaluate(
    query: Query,
    input_data: Data,
    backend: AvailableBackends = "pandas",
) -> Dict[Event, Any]:
    """Evaluates a query on data."""

    if isinstance(query, Event):
        events_to_compute = {query}

    elif isinstance(query, list):
        events_to_compute = set(query)

    elif isinstance(query, dict):
        raise NotImplementedError()

    else:
        # TODO: improve error message
        raise TypeError(
            f"schedule_graph query argument must be one of {Query}. Received"
            f" {type(query)}."
        )

    # get backend
    selected_backend = backends.BACKENDS[backend]
    event = selected_backend["event"]
    evaluate_schedule_fn = selected_backend["evaluate_schedule_fn"]
    read_csv_fn = selected_backend["read_csv_fn"]

    # calculate opeartor schedule
    schedule = get_operator_schedule(events_to_compute)

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

    return {event: outputs[event] for event in events_to_compute}


def get_operator_schedule(query: Set[Event]) -> List[base.Operator]:
    """Calculates which operators need to be executed in which order to
    compute a set of query events.
    Args:
        query: set of query events to be computed.
    Returns:
        ordered list of operators, such that the first operator should be
        computed before the second, second before the third, etc.
    """

    # TODO: add depth analysis for parallelization
    def visit(
        event: Event,
        pending_events: Set[Event],
        done_events: Set[Event],
        sorted_ops: List[base.Operator],
    ):
        if event in done_events:
            return
        if event in pending_events:
            raise RuntimeError(
                "Compute graph has at least one cycle - aborting."
            )

        # event pending - must wait for parent events
        pending_events.add(event)

        # get parent events
        parent_events = (
            {}
            if event.creator() is None
            else {
                parent_event
                for parent_event in event.creator().inputs().values()
            }
        )
        # recursion
        for parent_event in parent_events:
            visit(parent_event, pending_events, done_events, sorted_ops)

        # event cleared
        pending_events.remove(event)
        done_events.add(event)

        # add operator to schedule
        if event.creator() is not None:
            sorted_ops.append(event.creator())

    # events that have already been cleared for computation
    done_events = set()

    # events pending to be cleared for computation
    pending_events = set()

    # final operator schedule
    sorted_ops = []
    while not query.issubset(done_events):
        event = next(iter(query))
        visit(event, pending_events, done_events, sorted_ops)

    return sorted_ops
