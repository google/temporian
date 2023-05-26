from __future__ import annotations
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from temporian.implementation.numpy.data.event_set import (
    EventSet,
    IndexData,
    numpy_array_to_tp_dtype,
    normalize_timestamps,
    normalize_features,
)
from temporian.core.evaluation import evaluate
from temporian.core.operators.all_operators import add_index
from temporian.core.data.schema import Schema

# Array of values as feed by the user.
DataArray = Union[List[Any], np.ndarray]


def event_set(
    timestamps: DataArray,
    features: dict[str, DataArray],
    index_features: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> EventSet:
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

    print("@@@features:\n", features)
    print("@@@index_features:\n", index_features)

    features = {
        name: normalize_features(value) for name, value in features.items()
    }

    print("@@@normalized features:\n", features)

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
            features[feature_name] for feature_name in schema.feature_names()
        ],
        timestamps=timestamps,
        schema=schema,
    )
    evtset = EventSet(
        schema=schema,
        data={(): index_data},
    )

    if index_features:
        # Index the data
        input_node = evtset.node()
        output_node = add_index(input_node, index_to_add=index_features)
        evtset = evaluate(output_node, {input_node: evtset})
        assert isinstance(evtset, EventSet)

    evtset.name = name

    return evtset


def pd_dataframe_to_event_set(
    df: pd.DataFrame,
    index_names: Optional[List[str]] = None,
    timestamp_column: str = "timestamp",
    name: Optional[str] = None,
):
    """Creates an EventSet from a pandas DataFrame.

    Args:
        df: DataFrame to convert to an EventSet.
        index_names: Names of the DataFrame columns to be used as index for
            the event set. Defaults to [].
        timestamp_column: Name of the column containing the timestamps.
            Supported date types:
            `{np.datetime64, pd.Timestamp, datetime.datetime}`.
            Timestamps of these types are converted to UTC epoch float.
        is_sorted: If True, the DataFrame is assumed to be sorted by
            timestamp. If False, the DataFrame will be sorted by timestamp.
        name: Optional name for the EventSet.


    Returns:
        EventSet created from DataFrame.

    Raises:
        ValueError: If `index_names` or `timestamp_column` are not in `df`'s
            columns.
        ValueError: If a column has an unsupported dtype.

    Example:
        >>> import pandas as pd
        >>> from temporian.implementation.numpy.data.event_set import EventSet
        >>> df = pd.DataFrame(
        ...     data=[
        ...         [666964, 1.0, 740.0],
        ...         [666964, 2.0, 508.0],
        ...         [574016, 3.0, 573.0],
        ...     ],
        ...     columns=["product_id", "timestamp", "costs"],
        ... )
        >>> evset = EventSet.from_dataframe(df, index_names=["product_id"])
    """

    return event_set(
        timestamps=df[timestamp_column].to_numpy(copy=True),
        features=df.drop(columns=timestamp_column).to_dict("series"),
        index_features=index_names,
        name=name,
    )


def event_set_to_pd_dataframe(
    evtset: EventSet,
) -> pd.DataFrame:
    """Convert a EventSet to a pandas DataFrame.

    Returns:
        DataFrame created from EventSet.
    """

    timestamp_key = "timestamp"

    # Collect data into a dictionary.
    column_names = (
        evtset.schema.index_names()
        + evtset.schema.feature_names()
        + [timestamp_key]
    )
    data = {column_name: [] for column_name in column_names}
    for index_key, index_data in evtset._data.items():
        assert isinstance(index_key, tuple)

        # Timestamps
        data[timestamp_key].extend(index_data.timestamps)

        # Features
        for feature_name, feature in zip(
            evtset.schema.feature_names(), index_data.features
        ):
            data[feature_name].extend(feature)

        # Indexes
        for i, index_name in enumerate(evtset.schema.index_names()):
            data[index_name].extend([index_key[i]] * len(index_data.timestamps))

    return pd.DataFrame(data)
