from __future__ import annotations
from typing import Any, List, Optional, Union, Dict

import numpy as np

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
DataArray = Union[List[Any], np.ndarray, "pandas.Series"]


# Note: Keep the documentation about supported types in sync with
# "normalize_timestamp" and "normalize_features".
def event_set(
    timestamps: DataArray,
    features: Optional[Dict[str, DataArray]] = None,
    index_features: Optional[List[str]] = None,
    name: Optional[str] = None,
    is_unix_timestamp: Optional[bool] = None,
    same_sampling_as: Optional[EventSet] = None,
) -> EventSet:
    """Creates an event set from arrays (list, numpy, pandas).

    Usage example:

        ```python
        # Creates an event set with 4 timestamps and 3 features.
        evset = tp.event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "feature_1": [0.5, 0.6, math.nan, 0.9],
                "feature_2": ["red", "blue", "red", "blue"],
                "feature_3": [10, -1, 5, 5],
            },
        )

        # Creates an event set with an index.
        evset = tp.event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "feature_1": [0.5, 0.6, math.nan, 0.9],
                "feature_2": ["red", "blue", "red", "blue"],
            },
            index_features=["feature_2"],
        )

        # Create an evet set with datetimes.
        from datetime import datetime
        evset = tp.event_set(
            timestamps=[datetime(2015, 1, 1), datetime(2015, 1, 2)],
            features={
                "feature_1": [0.5, 0.6],
                "feature_2": ["red", "blue"],
            },
            index_features=["feature_2"],
        )
        ```

    Supported values for `timestamps`:

        - List of int, float, str, bytes and datetime.
        - Numpy arrays of int{32, 64}, float{32, 64}, str_, string_ / bytes_,
           Numpy datetime64, and object containing "str".
        - Pandas series of int{32, 64}, float{32, 64}, Pandas Timestamp.

    String timestamps are interpreted as ISO 8601 datetime.

    Supported values for `features`:

        - List of int, float, str, bytes, bool, and datetime.
        - Numpy arrays of int{32, 64}, float{32, 64}, str_, string_ / bytes_,
            Numpy datetime64, or object containing "str".
        - Pandas series of int{32, 64}, float{32, 64}, Pandas Timestamp.

    Date / datetime features are converted to int64 unix times.
    NaN for float-like features are interpreted as missing values.

    Args:
        timestamps: Array of timestamps values.
        features: Dictionary of feature names to feature values. Feature
            and timestamp arrays must be of the same length.
        index_features: Names of the features to use as index. If empty
            (default), the data is not indexed. Only integer and string features
            can be used as index.
        name: Optional name of the event set. Used for debugging, and
            graph serialization.
        is_unix_timestamp: Whether the timestamps correspond to unix time. Unix
            times are required for calendar operators. If `None` (default),
            timestamps are interpreted as unix times if the `timestamps`
            argument is an array of date or date-like object.
        same_sampling_as: If set, the new event set is cheched and tagged as
            having the same sampling as `same_sampling_as`. Some operators,
            such as `tp.filter`, require their inputes to have the same
            sampling.

    Returns:
        An event set.
    """

    if features is None:
        features = {}

    features = {
        name: normalize_features(value) for name, value in features.items()
    }

    # Convert timestamps to expected type.
    timestamps, auto_is_unix_timestamp = normalize_timestamps(timestamps)

    if not np.all(timestamps[:-1] <= timestamps[1:]):
        order = np.argsort(timestamps, kind="mergesort")
        timestamps = timestamps[order]
        features = {name: value[order] for name, value in features.items()}

    if is_unix_timestamp is None:
        is_unix_timestamp = auto_is_unix_timestamp
    assert isinstance(is_unix_timestamp, bool)

    # Infer the schema
    schema = Schema(
        features=[
            (feature_key, numpy_array_to_tp_dtype(feature_key, feature_data))
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

    if same_sampling_as is not None:
        evtset.schema.check_compatible_index(
            same_sampling_as.schema,
            label="the new event set and `same_sampling_as`",
        )

        if evtset.data.keys() != same_sampling_as.data.keys():
            raise ValueError(
                "The new event set and `same_sampling_as` have the same index,"
                " but different index values. Both should have the same index"
                " keys to have the same sampling."
            )

        for key, same_sampling_as_value in same_sampling_as.data.items():
            if not np.all(
                evtset.data[key].timestamps == same_sampling_as_value.timestamps
            ):
                raise ValueError(
                    "The new event set and `same_sampling_as` have different"
                    f" timestamps values for the index={key!r}. The timestamps"
                    " should be equal for both to have the same sampling."
                )

            # Discard the new timestamps arrays.
            evtset.data[key].timestamps = same_sampling_as_value.timestamps

        evtset.node()._sampling = same_sampling_as.node().sampling_node

    return evtset


def pd_dataframe_to_event_set(
    df: "pandas.DataFrame",
    index_names: Optional[List[str]] = None,
    timestamp_column: str = "timestamp",
    name: Optional[str] = None,
    same_sampling_as: Optional[EventSet] = None,
) -> EventSet:
    """Converts a Pandas DataFrame into an Event Set.

    TODO: Rename argument `index_names` to `index_features`.

    The column `timestamp_column` (default to "timestamp") contains the
    timestamps. Columns `index_names` (default to `None`, equivalent to `[]`),
    contains the index. The remaining columns are converted into features.

    See `tp.event_set` for the list of supported timestamp and feature types.

    Usage example:

    ```
    import pandas as pd
    df = pd.DataFrame(
        data=[
            [1.0, 5, "A"],
            [2.0, 6, "A"],
            [3.0, 7, "B"],
        ],
        columns=["timestamp", "feature_1", "feature_2"],
    )

    evset = tp.pd_dataframe_to_event_set(df, index_names=["feature_2"])
    ```

    Args:
        df: A non indexed Pandas dataframe.
        index_names: Names of the features to use as index. If empty
            (default), the data is not indexed. Only integer and string features
            can be used as index.
        timestamp_column: Name of the column containing the timestamps. See
            `tp.event_set`for the list of supported timestamp types.
        name: Optional name of the event set. Used for debugging, and
            graph serialization.
        same_sampling_as: If set, the new event set is cheched and tagged as
            having the same sampling as `same_sampling_as`. Some operators,
            such as `tp.filter`, require their inputes to have the same
            sampling.

    Returns:
        An event set.

    Raises:
        ValueError: If `index_names` or `timestamp_column` are not in `df`'s
            columns.
        ValueError: If a column has an unsupported dtype.
    """

    feature_dict = df.drop(columns=timestamp_column).to_dict("series")

    return event_set(
        timestamps=df[timestamp_column].to_numpy(copy=True),
        features={k: v.to_numpy(copy=True) for k, v in feature_dict.items()},
        index_features=index_names,
        name=name,
        same_sampling_as=same_sampling_as,
    )


def event_set_to_pd_dataframe(
    evtset: EventSet,
) -> "pandas.DataFrame":
    """Convert an EventSet to a pandas DataFrame.

    Returns:
        DataFrame created from EventSet.
    """

    import pandas as pd

    timestamp_key = "timestamp"

    # Collect data into a dictionary.
    column_names = (
        evtset.schema.index_names()
        + evtset.schema.feature_names()
        + [timestamp_key]
    )
    data = {column_name: [] for column_name in column_names}
    for index_key, index_data in evtset.data.items():
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
