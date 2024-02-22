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

"""Utilities for converting EventSets to numpy arrays and viceversa."""

import numpy as np
from numpy import ndarray

from typing import Dict
from temporian.implementation.numpy.data.event_set import EventSet


def to_numpy(
    evset: EventSet,
    timestamp_to_datetime: bool = True,
    timestamps: bool = True,
) -> Dict[str, ndarray]:
    """Converts an [`EventSet`][temporian.EventSet] to a flattened dictionary with
    numpy arrays.

    Usage example:
        ```python
        >>> from datetime import datetime

        >>> evset = tp.event_set(
        ...     timestamps=[datetime(2024, 2, 22), datetime(2015, 2, 23)],
        ...     features={
        ...         "feature_1": [0.5, 0.6],
        ...         "my_index": ["red", "red"],
        ...     },
        ...     indexes=["my_index"],
        ... )

        # datetime64[s] if they were created as datetimes, otherwhise they are
        # floats
        >>> res = tp.to_numpy(evset)
        >>> res
        {
            "timestamps": ['2023-11-08', '2023-11-29'],
            "feature_1": [0.5, 0.6],
            "my_index": ["red", "red"],
        }
        ```
    Args:
        evset: input event set.
        timestamp_to_datetime: If true, cast Temporian timestamps to datetime64
            when is_unix_timestamp is set to True.
        timestamps: If true, the timestamps are included as a column.

    Returns:
        object with numpy arrays created from EventSet.
    """
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

    dst = {k: np.concatenate(v) for k, v in dst.items()}
    return dst
