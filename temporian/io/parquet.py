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

"""Utilities for reading and saving EventSets from/to disk."""

from typing import List, Optional

from temporian.implementation.numpy.data.event_set import EventSet
from temporian.io.pandas import from_pandas, to_pandas
from temporian.utils.typecheck import typecheck


@typecheck
def from_parquet(
    path: str,
    timestamps: str = "timestamp",
    indexes: Optional[List[str]] = None,
    **kwargs,
) -> EventSet:
    """Reads an [`EventSet`][temporian.EventSet] from a parquet file.

    Example:
        ```python
        >>> temp_file = str(tmp_dir / "temporal_data.parquet")
        >>> og_eventset = tp.event_set(timestamps=[1,], features={"f1": [0.1]})
        >>> tp.to_parquet(og_eventset, temp_file)
        >>> evset = tp.from_parquet(temp_file)
        >>> evset
        indexes: []
        features: [('f1', float64)]
        events:
            (1 events):
                timestamps: [1.]
                'f1': [0.1]
        ...

        ```

    Args:
        path: Path to the file.
        timestamps: Name of the column to be used as timestamps for the
            EventSet.
        indexes: Names of the columns to be used as indexes for the EventSet.
            If None, a flat EventSet will be created.

    Returns:
        EventSet read from file.
    """

    import pandas as pd

    if indexes is None:
        indexes = []

    df = pd.read_parquet(path, **kwargs)
    return from_pandas(df, indexes=indexes, timestamps=timestamps)


@typecheck
def to_parquet(evset: EventSet, path: str, **kwargs):
    """Saves an [`EventSet`][temporian.EventSet] to a CSV file.

    Example:
        ```python
        >>> output_path = str(tmp_dir / "output_data.parquet")
        >>> evset = tp.event_set(timestamps=[1,], features={"f1": [0.1]})
        >>> tp.to_parquet(evset, output_path)

        ```

    Args:
        evset: EventSet to save.
        path: Path to the file.
    """
    # TODO: include indexes in the file's metadata
    # so that they can be recovered automatically
    df = to_pandas(evset)
    df.to_parquet(path, **kwargs)
