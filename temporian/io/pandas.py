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

"""Utilities for converting EventSets to pandas DataFrames and viceversa."""

import numpy as np

from typing import List, Optional
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.io import event_set
from temporian.core.data.dtype import DType


# TODO: Rename argument `index_names` to `index_features`.
def from_pandas(
    df: "pandas.DataFrame",
    indexes: Optional[List[str]] = None,
    timestamps: str = "timestamp",
    name: Optional[str] = None,
    same_sampling_as: Optional[EventSet] = None,
) -> EventSet:
    """Converts a Pandas DataFrame into an [`EventSet`][temporian.EventSet].

    The column `timestamps` (defaults to "timestamp") contains the
    timestamps. Columns `indexes` (default to `None`, equivalent to `[]`),
    contains the indexes. The remaining columns are converted into features.

    See [`tp.event_set()`][temporian.event_set] for the list of supported
    timestamp and feature types.

    Usage example:
        ```python
        >>> df = pd.DataFrame(
        ...     data=[
        ...         [1.0, 5, "A"],
        ...         [2.0, 6, "A"],
        ...         [3.0, 7, "B"],
        ...     ],
        ...     columns=["timestamp", "feature_1", "feature_2"],
        ... )
        >>> evset = tp.from_pandas(df, indexes=["feature_2"])

        ```

    Args:
        df: A non indexed Pandas dataframe.
        indexes: Names of the columns to use as indexes. If empty
            (default), the data is not indexed. Only integer and string columns
            can be used as indexes.
        timestamps: Name of the column containing the timestamps. See
            [`tp.event_set()`][temporian.event_set] for the list of supported
            timestamp types.
        name: Optional name of the EventSet. Used for debugging, and
            graph serialization.
        same_sampling_as: If set, the new EventSet is cheched and tagged as
            having the same sampling as `same_sampling_as`. Some operators,
            such as [`EventSet.filter()`][temporian.EventSet.filter], require
            their inputs to have the same sampling.

    Returns:
        An EventSet.

    Raises:
        ValueError: If `indexes` or `timestamps` are not in `df`'s
            columns.
        ValueError: If a column has an unsupported dtype.
    """

    feature_dict = df.drop(columns=timestamps).to_dict("series")

    return event_set(
        timestamps=df[timestamps].to_numpy(),
        features={k: v.to_numpy() for k, v in feature_dict.items()},
        indexes=indexes,
        name=name,
        same_sampling_as=same_sampling_as,
    )


def to_pandas(
    evset: EventSet,
    tp_string_to_pd_string: bool = True,
    timestamp_to_datetime: bool = True,
    timestamps: bool = True,
) -> "pandas.DataFrame":
    """Converts an [`EventSet`][temporian.EventSet] to a pandas DataFrame.

    Usage example:
        ```python
        >>> from datetime import datetime

        >>> evset = tp.event_set(
        ...     timestamps=[datetime(2015, 1, 1), datetime(2015, 1, 2)],
        ...     features={
        ...         "feature_1": [0.5, 0.6],
        ...         "my_index": ["red", "red"],
        ...     },
        ...     indexes=["my_index"],
        ... )

        # Indices are not set as dataframe's indices. Timestamps are exported as
        # datetime64[s] if they were created as datetimes, otherwhise they are
        # floats
        >>> df = tp.to_pandas(evset)
        >>> df
        my_index   feature_1     timestamp
        0       red        0.5  2015-01-01
        1       red        0.6  2015-01-02

        # Set index/date manually in pandas
        >>> df.set_index("my_index")
                    feature_1  timestamp
        my_index
        red         0.5     2015-01-01
        red         0.6     2015-01-02

        ```

    Args:
        evset: Input event set.
        tp_string_to_pd_string: If true, cast Temporian strings (equivalent to
            np.string_ or np.bytes) to Pandas strings (equivalent to np.str_).
        timestamp_to_datetime: If true, cast Temporian timestamps to datetime64
            when is_unix_timestamp is set to True.
        timestamps: If true, the timestamps are included as a column.

    Returns:
        DataFrame created from EventSet.
    """

    import pandas as pd

    timestamp_key = "timestamp"

    index_names = evset.schema.index_names()
    feature_names = evset.schema.feature_names()

    column_names = index_names + feature_names
    if timestamps:
        column_names += [timestamp_key]

    dst = {column_name: [] for column_name in column_names}
    for index, data in evset.data.items():
        assert isinstance(index, tuple)

        if timestamps:
            # Timestamps
            if evset.schema.is_unix_timestamp and timestamp_to_datetime:
                dst[timestamp_key].append(
                    data.timestamps.astype("datetime64[s]")
                )
            else:
                dst[timestamp_key].append(data.timestamps)

        # Features
        for feature_name, feature in zip(feature_names, data.features):
            dst[feature_name].append(feature)

        # Indexes
        num_timestamps = len(data.timestamps)
        for index_name, index_item in zip(index_names, index):
            dst[index_name].append(np.repeat(index_item, num_timestamps))

    dst = {k: np.concatenate(v) for k, v in dst.items()}

    if tp_string_to_pd_string:
        for feature in evset.schema.features:
            if feature.dtype == DType.STRING:
                dst[feature.name] = dst[feature.name].astype(str)
        for index in evset.schema.indexes:
            if index.dtype == DType.STRING:
                dst[index.name] = dst[index.name].astype(str)

    return pd.DataFrame(dst)
