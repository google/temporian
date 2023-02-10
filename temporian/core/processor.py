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

"""Processor module."""

from typing import List, Set, Dict, Union, Optional

from collections import defaultdict

from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base

MultipleEventArg = Union[Dict[str, Event], List[Event], Event]


def normalize_multiple_event_arg(src: MultipleEventArg) -> Dict[str, Event]:
    """Normalize an event / collection of events into a dictionary of events."""

    save_src = src

    if isinstance(src, Event):
        src = [src]

    if isinstance(src, list):
        new_src = {}
        for event in src:
            if event.name() is None:
                raise ValueError(
                    "Input / output event or list events need to be named "
                    'with "set_name(...)". Alternatively, provide a '
                    "dictionary of events."
                )
            new_src[event.name()] = event
        src = new_src

    if not isinstance(src, dict):
        raise ValueError(
            f'Unexpected event(s) "{save_src}". Expecting dict of events, '
            "list of events, or a single event."
        )
    return src


class Preprocessor(object):
    """A set of operators, events, features and samplings."""

    def __init__(self):
        self._operators: Set[base.Operator] = set()
        self._features: Set[Feature] = set()
        self._events: Set[Event] = set()
        self._samplings: Set[Sampling] = set()
        self._inputs: Dict[str, Event] = {}
        self._outputs: Dict[str, Event] = {}

    def samplings(self) -> Set[Sampling]:
        return self._samplings

    def features(self) -> Set[Feature]:
        return self._features

    def operators(self) -> Set[base.Operator]:
        return self._operators

    def events(self) -> Set[Event]:
        return self._events

    def add_operator(self, operator: base.Operator) -> None:
        self._operators.add(operator)

    def add_sampling(self, sampling: Sampling) -> None:
        self._samplings.add(sampling)

    def add_feature(self, feature: Feature) -> None:
        self._features.add(feature)

    def add_event(self, event: Event) -> None:
        self._events.add(event)

    def set_inputs(self, inputs: Dict[str, Event]) -> None:
        self._inputs = inputs

    def set_outputs(self, outputs: Dict[str, Event]) -> None:
        self._outputs = outputs

    def inputs(self) -> Dict[str, Event]:
        return self._inputs

    def outputs(self) -> Dict[str, Event]:
        return self._outputs

    def input_features(self) -> Set[Feature]:
        return {
            feature
            for event in self.inputs().values()
            for feature in event.features()
        }

    def input_samplings(self) -> Set[Sampling]:
        return {event.sampling() for event in self.inputs().values()}

    def __repr__(self):
        s = "Preprocessor\n============\n"

        def p(title, elements):
            nonlocal s
            s += f"{title} ({len(elements)}):\n"
            for e in elements:
                s += f"\t{e}\n"
            s += "\n"

        p("Operators", self.operators())
        p("Features", self.features())
        p("Samplings", self.samplings())
        p("Events", self.events())

        def p2(title, dictionary):
            nonlocal s
            s += f"{title} ({len(dictionary)}):\n"
            for k, v in dictionary.items():
                s += f"\t{k}:{v}\n"
            s += "\n"

        p2("Inputs", self.inputs())
        p2("Output", self.outputs())
        return s


def infer_processor(
    inputs: Optional[Dict[str, Event]],
    outputs: Dict[str, Event],
) -> Preprocessor:
    """Extracts all the objects between the outputs and inputs events.

    Fails if some inputs are missing.

    Args:
        inputs: Input events. If None, the inputs is infered. In this
            case, input event have to be named.
        outputs: Output events.

    Returns:
      A preprocessor.
    """

    # The following algorithm lists all the events between the output and
    # input events. Informally, the algorithm works as follow:
    #
    # pending_event <= use outputs
    # done_event <= empty
    #
    # While pending event not empty:
    #   Extract an event from pending_event
    #   if event is a provided input event
    #       continue
    #   if event has not creator
    #       record this event for future error / input inference
    #       continue
    #   Adds all the input events of event's creator op to the pending list

    p = Preprocessor()
    p.set_outputs(outputs)

    # The next event to process. Events are processed from the outputs to
    # the inputs.
    pending_events: Set[Event] = set()
    pending_events.update(outputs.values())

    # Index the input event for fast retrieval
    input_events: Set[Event] = {}

    if inputs is not None:
        p.set_inputs(inputs)
        input_events = set(inputs.values())

    # Features already processed.
    done_events: Set[Event] = set()

    # List of the missing events. They will be used to infer the input features
    # (if infer_inputs=True), or to raise an error (if infer_inputs=False).
    missing_events: Set[Event] = set()

    while pending_events:
        # Select an event to process.
        event = next(iter(pending_events))
        pending_events.remove(event)
        assert event not in done_events

        p.add_event(event)

        if event in input_events:
            # The feature is provided by the user.
            continue

        if event.creator() is None:
            # The event does not have a source.
            missing_events.add(event)
            continue

        # Record the operator.
        p.add_operator(event.creator())

        # Add the parent events to the pending list.
        for input_event in event.creator().inputs().values():

            if input_event in done_events:
                # Already processed.
                continue

            pending_events.add(input_event)

        # Record the operator outputs. While the user did not request
        # them, they will be created (and so, we need to track them).
        for output_event in event.creator().outputs().values():
            p.add_event(output_event)

    if inputs is None:
        # Infer the inputs
        infered_inputs: Dict[str, Event] = {}
        for event in missing_events:
            if event.name() is None:
                raise ValueError(f"Cannot infer input on unnamed event {event}")
            infered_inputs[event.name()] = event
        p.set_inputs(infered_inputs)

    else:
        # Fail if not all events are sourced.
        if missing_events:
            raise ValueError(
                "One of multiple events are required but "
                f"not provided as input:\n {missing_events}"
            )

    # Record all the features and samplings.
    for e in p.events():
        p.add_sampling(e.sampling())
        for f in e.features():
            p.add_feature(f)

    return p
