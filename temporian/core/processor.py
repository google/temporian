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
    infer_inputs: bool = False,
) -> Preprocessor:
    """Create a self contained processor.

    The processor contains all the features, samplings, events and operators
    in between "inputs" and "outputs". The processor contains all the outputs
    of all necessary operators (even it only part of those outputs are used).

    Fails if some inputs are missing.

    Args:
      inputs: Dictionary of available inputs.
      outputs: Dictionary of requested outputs.
    infer_inputs: If true, infers the inputs.

    Returns:
      A preprocessor.
    """

    p = Preprocessor()
    p.set_outputs(outputs)

    # The following algorithm lists all the intermediate and missing input
    # features of a computation graph.
    #
    # The algorithm works as follows:
    #
    # pending_features <= List of outputs
    # done_features <= empty
    #
    # While pending feature not empty:
    #   Extract a feature from pending_features
    #   if feature is part of the provided input features => Continue
    #   if feature has a creator operator
    #     Adds all the features, of all the inputs of this operator to
    #     pending_features. Skip the features already in pending_features or
    #     done_features.
    #   else:
    #     add feature to the list of missing input features (will raise an error)

    # List the events containing each possible feature:
    # Used for the automatic input inference.
    features_to_src_event: Dict[Feature, List[Event]] = defaultdict(lambda: [])

    pending_features: Set[Feature] = set()
    for output_event in outputs.values():
        pending_features.update(output_event.features())
        p.add_sampling(output_event.sampling())
        for feature in output_event.features():
            features_to_src_event[feature].append(pending_features)

    input_features: Set[Feature] = set()
    if not infer_inputs:
        p.set_inputs(inputs)
        for input_event in inputs.values():
            input_features.update(input_event.features())
            p.add_sampling(input_event.sampling())
    else:
        infered_inputs: Dict[str, Event] = {}

    done_features: Set[Feature] = set()

    # Text description of the missing features.
    missing_features: Set[str] = set()

    while pending_features:
        # Select a feature from pending_features.
        feature = next(iter(pending_features))
        pending_features.remove(feature)

        p.add_feature(feature)

        assert feature not in done_features

        if feature in input_features:
            # The feature is provided by the user.
            continue

        if feature.creator() is None:
            # The feature is missing.

            if infer_inputs:
                # Infer the inputs
                if feature not in features_to_src_event:
                    raise ValueError(
                        f"Cannot determine the input of {feature}. "
                        "No event found."
                    )
                if len(features_to_src_event[feature]) > 1:
                    raise ValueError(
                        f"Cannot determine the input of {feature}. "
                        "Multiple potential events found."
                    )

                infered_input = list(features_to_src_event[feature])[0]
                if infered_input.name() is None:
                    raise ValueError(
                        f"Infered input without a name: {infered_input}."
                    )
                infered_inputs[infered_input.name()] = infered_input
                p.add_sampling(feature.sampling())
            else:
                missing_features.add(repr(feature))

        else:
            p.add_operator(feature.creator())
            p.add_sampling(feature.sampling())

            # Add the parent features.
            for input_event in feature.creator().inputs().values():
                p.add_event(input_event)
                for input_feature in input_event.features():
                    if input_feature in done_features:
                        continue
                    if input_feature in pending_features:
                        continue
                    pending_features.add(input_feature)

                    if infer_inputs and input_feature.creator() is None:
                        features_to_src_event[input_feature].append(input_event)

            # Make sure that all operator outputs are listed.
            for output_event in feature.creator().outputs().values():
                p.add_event(output_event)
                for output_feature in output_event.features():
                    p.add_feature(output_feature)

    if missing_features:
        raise ValueError(f"Missing input features: {missing_features}")

    if infer_inputs:
        p.set_inputs(infered_inputs)

    return p
