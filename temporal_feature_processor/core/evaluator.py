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

from typing import Dict, List, Literal, Set, Union

import pandas as pd

from temporal_feature_processor.core import backends
from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.core.data.feature import Feature
from temporal_feature_processor.core.operators import base

Query = Union[Event, Dict[str, Event], List[Event]]

Data = Union[pd.DataFrame, Dict[str, pd.DataFrame], List[pd.DataFrame]]

Result = Data

AvailableBackends = Literal[tuple(backends.BACKENDS.keys())]


def evaluate(
    query: Query,
    data: Data,
    backend: AvailableBackends = "pandas",
) -> Result:
  """Evaluates a query on data."""

  if isinstance(query, Event):
    events_to_compute = [query]

  elif isinstance(query, list):
    events_to_compute = query

  elif isinstance(query, dict):
    events_to_compute = list(query.values())

  else:
    # TODO: improve error message
    raise TypeError(
        f"schedule_graph query argument must be one of {Query}. Received {type(query)}."
    )

  # get features from events
  features_to_compute = [
      feature for event in events_to_compute for feature in event.features()
  ]

  # get backend
  evaluate_schedule_fun = backends.BACKENDS[backend]

  # calculate opeartor schedule
  schedule = get_operator_schedule(features_to_compute)

  # evaluate schedule
  outputs = evaluate_schedule_fun(data, schedule)

  return {event: outputs[event] for event in events_to_compute}


def get_operator_schedule(query: Set[Feature]) -> List[base.Operator]:
  # TODO: add depth calculation for parallelization

  operators_to_compute = []  # TODO: implement as ordered set
  visited_features = set()
  pending_features = list(query.copy())  # TODO: implement as ordered set
  while pending_features:
    feature = next(iter(pending_features))
    visited_features.add(feature)

    if feature.creator() is None:
      # is input feature
      pending_features.remove(feature)
      continue

    # required input features to compute this feature
    creator_input_features = {
        input_feature for input_event in feature.creator().inputs().values()
        for input_feature in input_event.features()
    }
    # check if all required input features have already been visited
    if creator_input_features.issubset(visited_features):
      # feature can be computed - remove it from pending_features
      pending_features.remove(feature)

      # add operator at the end of operators_to_compute
      if feature.creator() not in operators_to_compute:
        operators_to_compute.append(feature.creator())

      continue

    # add required input features at the beginning of pending_features
    pending_features = list(creator_input_features) + pending_features

  return operators_to_compute
