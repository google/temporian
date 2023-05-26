from __future__ import annotations
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from temporian.implementation.numpy.data.event_set import (
    EventSet,
    IndexData,
    numpy_array_to_tp_dtype,
    normalize_timestamps,
)
from temporian.core.evaluation import evaluate
from temporian.core.operators.all_operators import set_index
from temporian.core.data.schema import Schema

# Array of values as feed by the user.
DataArray = Union[List[Any], np.array]


def event_set(
    timestamps: DataArray,
    features: dict[str, DataArray],
    index_features: Optional[List[str]] = None,
    name: Optional[str] = None,
):
    """Creates an event set from raw data (e.g. python lists,  numpy arrays).

    Usage example:

        ```python
        evset = tp.evset = tp.event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "feature_1": [0.5, 0.6, math.nan, 0.9],
                "feature_2": ["red", "blue", "red", "blue"],
                "feature_3": [10, -1, 5, 5],
            },
        )
        ```

    Args:
        timestamps: Array of timestamps values. Can be a list of numpy array of
            float, integer, datetimes or dates.
        features: Dictionary of feature values.
        index_features: Names of the features in "features" to be used as index.
          If empty (default), the data is not indexed.
        name: Name of the node.

    Returns:
        An event set.
    """

    # Convert timestamps to expected type.
    timestamps, is_unix_timestamp = normalize_timestamps(timestamps)

    # Infer the schema
    schema = Schema(
        features=[
            (feature_key, numpy_array_to_tp_dtype(feature_data))
            for feature_key, feature_data in features.items()
        ],
        indexes=[],
        is_unix_timestamp=is_unix_timestamp,
    )

    # Shallow copy the data to temporian format
    index_data = IndexData(
        features=[
            features[feature_name] for feature_name in schema.feature_names
        ],
        timestamps=timestamps,
        schema=schema,
    )
    evtset = EventSet(
        schema=schema,
        data={
            (): index_data,
        },
    )

    if index_features:
        # Index the data
        input_node = evtset.node()
        output_node = set_index(input_node, feature_names=index_features)
        evtset = evaluate(output_node, {input_node: evtset})

    evtset.name = name

    return evtset
