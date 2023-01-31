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
from temporian.core.data.feature import Feature
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

    # get features from events
    features_to_compute = {
        feature
        for event in events_to_compute
        for feature in event.features()  # pytype: disable=attribute-error
    }
    # calculate opeartor schedule. Only using keys (operators) for now, discarding
    # depth
    schedule = get_operator_schedule(features_to_compute).keys()

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


def get_operator_schedule(query: Set[Feature]) -> Dict[base.Operator, int]:
    """Calculates which operators need to be executed in which order to
    compute a set of query features.

    Args:
        query: set of query features to be computed.

    Returns:
        Dict[base.Operator, int]: dictionary mapping operators to their respective
        depths in the compute graph. Depth is measured from bottom to top, i.e.
        0 depth corresponds to the output features (query). Operators with the
        same depth can be computed in parallel.
    """
    # start features. Depth initialized as 0
    feature_depth = [(feature, 0) for feature in query]

    # get all features and depths required to compute the query. One feature can
    # have multiple depths in this set (if it appears in more than one feature's
    # compute path at different depths)
    for feature, depth in feature_depth:
        this_depth = depth + 1
        for parent_feature in feature.parent_features():
            feature_depth.append((parent_feature, this_depth))

    # refine previous set - get max depth for each feature, which ensures we
    # compute the feature as soon as it's needed (most depth)
    feature_max_depth = {feature: 0 for feature, _ in feature_depth}
    for feature, depth in feature_depth:
        feature_max_depth[feature] = max(feature_max_depth[feature], depth)

    # sort features according to their max depth, from deepest to shallowest
    feature_sorted = dict(
        sorted(feature_max_depth.items(), key=lambda item: -1 * item[1])
    )

    # get operators from features
    operator_sorted = {
        feature.creator(): depth
        for feature, depth in feature_sorted.items()
        if feature.creator() is not None
    }
    return operator_sorted
