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

    Usage examples:

        ```python
        >>> # Creates an event set with 4 timestamps and 3 features.
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3, 4],
        ...     features={
        ...         "feature_1": [0.5, 0.6, np.nan, 0.9],
        ...         "feature_2": ["red", "blue", "red", "blue"],
        ...         "feature_3": [10, -1, 5, 5],
        ...     },
        ... )

        >>> # Creates an event set with an index.
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3, 4],
        ...     features={
        ...         "feature_1": [0.5, 0.6, np.nan, 0.9],
        ...         "feature_2": ["red", "blue", "red", "blue"],
        ...     },
        ...     index_features=["feature_2"],
        ... )

        >>> # Create an evet set with datetimes.
        >>> from datetime import datetime
        >>> evset = tp.event_set(
        ...     timestamps=[datetime(2015, 1, 1), datetime(2015, 1, 2)],
        ...     features={
        ...         "feature_1": [0.5, 0.6],
        ...         "feature_2": ["red", "blue"],
        ...     },
        ...     index_features=["feature_2"],
        ... )

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
    evset = EventSet(
        schema=schema,
        data={(): index_data},
    )

    if index_features:
        # Index the data
        input_node = evset.node()
        output_node = add_index(input_node, index_to_add=index_features)
        evset = evaluate(output_node, {input_node: evset})
        assert isinstance(evset, EventSet)

    evset.name = name

    if same_sampling_as is not None:
        evset.schema.check_compatible_index(
            same_sampling_as.schema,
            label="the new event set and `same_sampling_as`",
        )

        if evset.data.keys() != same_sampling_as.data.keys():
            raise ValueError(
                "The new event set and `same_sampling_as` have the same index,"
                " but different index values. Both should have the same index"
                " keys to have the same sampling."
            )

        for key, same_sampling_as_value in same_sampling_as.data.items():
            if not np.all(
                evset.data[key].timestamps == same_sampling_as_value.timestamps
            ):
                raise ValueError(
                    "The new event set and `same_sampling_as` have different"
                    f" timestamps values for the index={key!r}. The timestamps"
                    " should be equal for both to have the same sampling."
                )

            # Discard the new timestamps arrays.
            evset.data[key].timestamps = same_sampling_as_value.timestamps

        evset.node()._sampling = same_sampling_as.node().sampling_node

    return evset
