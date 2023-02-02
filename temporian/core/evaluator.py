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
        events_to_compute = [query]

    elif isinstance(query, list):
        events_to_compute = query

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


def get_operator_schedule(query: List[Event]) -> List[base.Operator]:
    # TODO: add depth calculation for parallelization
    # TODO: Make "query" a Set

    operators_to_compute = []  # TODO: implement as ordered set
    visited_events = set()
    pending_events = list(query.copy())  # TODO: implement as ordered set
    while pending_events:
        event = next(iter(pending_events))
        visited_events.add(event)

        if event.creator() is None:
            # is input event
            pending_events.remove(event)
            continue

        # required input events to compute this event
        creator_input_events = {
            input_event for input_event in event.creator().inputs().values()
        }

        # check if all required input events have already been visited
        if creator_input_events.issubset(visited_events):
            # feature can be computed - remove it from pending_events
            pending_events.remove(event)

            # add operator at the end of operators_to_compute
            if event.creator() not in operators_to_compute:
                operators_to_compute.append(event.creator())

            continue

        # add required input features at the beginning of pending_events
        pending_events = list(creator_input_events) + pending_events

    print(operators_to_compute)
    return operators_to_compute
