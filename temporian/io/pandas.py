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

"""Utility for reading an event set from disk."""

from typing import List, Optional
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.io import event_set


def from_pandas(
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

    evset = tp.from_pandas(df, index_names=["feature_2"])
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


def to_pandas(
    evtset: EventSet,
) -> "pandas.DataFrame":
    """Convert an EventSet to a pandas DataFrame.

    Returns:
        DataFrame created from EventSet.
    """

    import pandas as pd

    timestamp_key = "timestamp"

    # TODO: Optimize

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
