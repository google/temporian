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
from temporian.io.pandas import from_pandas, to_pandas


def from_csv(
    path: str,
    timestamp_column: str = "timestamp",
    index_names: Optional[List[str]] = None,
    sep: str = ",",
) -> EventSet:
    """Reads an EventSet from a file.

    Args:
        path: Path to the file.
        timestamp_column: Name of the column to be used as timestamps for the
            event set.
        index_names: Names of the columns to be used as index for the event set.
            If None, a flat event set will be created.
        sep: Separator to use.


    Returns:
        EventSet read from file.

    """

    import pandas as pd

    if index_names is None:
        index_names = []

    df = pd.read_csv(path, sep=sep)
    return from_pandas(
        df, index_names=index_names, timestamp_column=timestamp_column
    )


def to_csv(
    evset: EventSet,
    path: str,
    sep: str = ",",
    na_rep: Optional[str] = None,
    columns: Optional[List[str]] = None,
):
    """Saves an EventSet to a file.

    Args:
        evset: EventSet to save.
        path: Path to the file.
        sep: Separator to use.
        na_rep: Representation to use for missing values.
        columns: Columns to save. If `None`, saves all columns.
    """
    df = to_pandas(evset)
    df.to_csv(path, index=False, sep=sep, na_rep=na_rep, columns=columns)
