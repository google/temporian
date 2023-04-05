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

"""Utility for reading an event from disk."""

from typing import List
import pandas as pd
from temporian.implementation.numpy.data.event import NumpyEvent


def read_event(
    path: str,
    timestamp_column: str,
    index_names: List[str] = None,
    sep: str = ",",
) -> NumpyEvent:
    """Reads a NumpyEvent from a file.

    Args:
        path: Path to the file.
        timestamp_column: Name of the column to be used as timestamps for the
            event.
        index_names: Names of the columns to be used as index for the event.
            If None, a flat event will be created.
        sep: Separator to use.


    Returns:
        NumpyEvent: NumpyEvent read from file.

    """
    if index_names is None:
        index_names = []

    df = pd.read_csv(path, sep=sep)
    return NumpyEvent.from_dataframe(
        df, index_names=index_names, timestamp_column=timestamp_column
    )
