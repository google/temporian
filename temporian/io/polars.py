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

"""Utilities for converting EventSets to pandas DataFrames and vice versa."""


from typing import List, Optional
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.io import event_set
from temporian.core.data.dtype import DType


def from_polars(
    df: "polars.DataFrame",
    timestamps: str = "timestamp",
    indexes: Optional[
        List[str]
    ] = None,  # Clarification: Temporian indexes, not Polars
    name: Optional[str] = None,
    same_sampling_as: Optional[EventSet] = None,
) -> EventSet:
    """Converts a Polars DataFrame into an EventSet.

    See [`tp.event_set()`][temporian.event_set] for the list of supported
    timestamp and feature types.

    Usage example:
        ```python
        >>> df = pl.DataFrame(
                {
                    "product_id": [666964, 666964, 574016, 574016],
                    "timestamp": [1.0, 2.0, 3.0, None],
                    "costs": [740.0, 508.0, 573.0, 573.0],
                }
            )
        >>> evset = tp.from_polars(df, indexes=["product_id"])

        ```

    Args:
        df: A Polars DataFrame.
        timestamps: Column name for timestamps.
        indexes: Names of the columns to use as Temporian indexes. If None, the data is not indexed.
        name: Optional name for the EventSet.
        same_sampling_as: Ensures the new EventSet has the same sampling as this EventSet.

    Returns:
        An instance of EventSet.

    Note:
        The function attempts to minimize data copying but will copy if required for compatibility.
    """
    if timestamps not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamps}' not found in the DataFrame."
        )

    # Extract timestamps, allowing copy if necessary for compatibility
    timestamps_array = df.get_column(timestamps).to_numpy(allow_copy=True)

    # Prepare features, allowing copy if necessary
    feature_columns = [col for col in df.columns if col != timestamps]
    feature_dict = {
        col: df.get_column(col).to_numpy(allow_copy=True)
        for col in feature_columns
    }

    return event_set(
        timestamps=timestamps_array,
        features=feature_dict,
        indexes=indexes,
        name=name,
        same_sampling_as=same_sampling_as,
    )


def to_polars(
    evset: EventSet,
    timestamp_to_datetime: bool = True,
    timestamps: bool = True,
    tp_string_to_pl_string: bool = True,
) -> "pl.DataFrame":
    """Converts an  [`EventSet`][temporian.EventSet] to a Polars DataFrame.

    Usage example:
        ```python
        >>> from datetime import datetime

        >>> evset = tp.event_set(
        ...     timestamps=[datetime(2015, 1, 1), datetime(2015, 1, 2)],
        ...     features={
        ...         "feature_1": [0.5, 0.6],
        ...         "my_index": ["red", "yellow"],
        ...    },
        ...    indexes=["my_index"],
        ... )

        >>> df = tp.to_polars(evset)
        >>> df
        shape: (2, 3)
        ┌──────────┬───────────┬─────────────────────┐
        │ my_index ┆ feature_1 ┆ timestamp           │
        │ ---      ┆ ---       ┆ ---                 │
        │ str      ┆ f64       ┆ datetime[μs]        │
        ╞══════════╪═══════════╪═════════════════════╡
        │ red      ┆ 0.5       ┆ 2015-01-01 00:00:00 │
        │ red      ┆ 0.6       ┆ 2015-01-02 00:00:00 │
        └──────────┴───────────┴─────────────────────┘
        ```

    Args:
        evset: Input event set.
        timestamp_to_datetime: If true, convert epoch timestamps to Polars Date objects.
        timestamps: If true, include the timestamps as a column in the DataFrame.
        tp_string_to_pl_string: If true, cast Temporian strings to Polars Object.

    Returns:
        A Polars DataFrame created from EventSet.
    """

    import polars as pl

    timestamp_key = "timestamp"

    index_names = evset.schema.index_names()
    feature_names = evset.schema.feature_names()

    column_names = index_names + feature_names
    if timestamps:
        column_names += [timestamp_key]

    # Initialize an empty dictionary to hold column data
    data_dict = {column_name: [] for column_name in column_names}

    for index, data in evset.data.items():
        assert isinstance(index, tuple)

        if timestamps:
            timestamps_data = data.timestamps
            if evset.schema.is_unix_timestamp and timestamp_to_datetime:
                # Convert Unix timestamps to Polars datetime objects
                # Assuming timestamps_data is a list of integers representing Unix timestamps in seconds
                datetime_series = pl.from_epoch(
                    pl.Series(timestamps_data), time_unit="s"
                )

                data_dict[timestamp_key].extend(datetime_series)
            else:
                data_dict[timestamp_key].extend(timestamps_data)

        # Features
        for feature_name, feature in zip(feature_names, data.features):
            data_dict[feature_name].extend(feature)

        # Indexes
        num_timestamps = len(data.timestamps)
        for index_name, index_item in zip(index_names, index):
            data_dict[index_name].extend([index_item] * num_timestamps)

    # Concatenate lists of values for each column
    for col_name, col_data in data_dict.items():
        data_dict[col_name] = pl.Series(col_data)

    if tp_string_to_pl_string:
        for feature in evset.schema.features:
            if feature.dtype == DType.STRING:
                data_dict[feature.name] = data_dict[feature.name].cast(pl.Utf8)
        for index in evset.schema.indexes:
            if index.dtype == DType.STRING:
                data_dict[index.name] = data_dict[index.name].cast(pl.Utf8)

    return pl.DataFrame(data_dict)
