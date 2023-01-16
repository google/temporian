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

from typing import List, Set

from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.core.data.feature import Feature
from temporal_feature_processor.core.data.sampling import Sampling
from temporal_feature_processor.core.operators import base


class Preprocessor(object):
  """A set of operators, events, features and samplings."""

  def __init__(self):
    self._operators: Set[base.Operator] = set()
    self._features: Set[Feature] = set()

  def samplings(self) -> Set[Sampling]:
    samplings = set()
    for feature in self._features:
      if feature.sampling() is not None:
        samplings.add(feature.sampling())
    return samplings

  def events(self) -> Set[Event]:
    events = set()
    for operator in self._operators:
      for event in operator.inputs().values():
        events.add(event)
      for event in operator.outputs().values():
        events.add(event)
    return events

  def features(self):
    return self._features

  def operators(self):
    return self._operators

  def __repr__(self):
    s = "Preprocessor\n============\n"

    def p(title, elements):
      nonlocal s
      s += f"{title} ({len(elements)}):\n"
      for e in elements:
        s += f"\t{e}\n"
      s += "\n"

    p("Operators", self._operators)
    p("Features", self._features)
    p("Samplings", self.samplings())
    p("Events", self.events())
    return s


def infer_processor(inputs: List[Event], outputs: List[Event]) -> Preprocessor:
  """Create a self contained processor.

  Fails if some inputs are missing.

  Args:
    inputs: List of available inputs.
    outputs: List of requested outputs.

  Returns:
    A preprocessor.
  """

  p = Preprocessor()

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

  pending_features: Set[Feature] = set()
  for output_event in outputs:
    pending_features.update(output_event.features())

  input_features: Set[Feature] = set()
  for input_event in inputs:
    input_features.update(input_event.features())

  done_features: Set[Feature] = set()

  # Text description of the missing features
  missing_features: Set[str] = set()

  while pending_features:
    # Select a feature from pending_features
    feature = next(iter(pending_features))
    pending_features.remove(feature)

    p.features().add(feature)

    assert feature not in done_features

    if feature in input_features:
      # The feature is provided by the user.
      continue

    if feature.creator() is None:
      # The feature is missing.
      missing_features.add(repr(feature))

    if feature.creator().is_placeholder():
      # The user is expected to see the input of placeholders.
      missing_features.add(
          f"{repr(feature)} from placeholder {feature.creator()}"
      )

    else:
      p.operators().add(feature.creator())
      # Add the parent features
      for input_event in feature.creator().inputs().values():
        for input_feature in input_event.features():
          if input_feature in done_features:
            continue
          if input_feature in pending_features:
            continue
          pending_features.add(input_feature)

  if missing_features:
    raise ValueError(f"Missing input features: {missing_features}")

  return p
