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

"""Utility for saving an event set to disk."""

from typing import List, Optional

from temporian.implementation.numpy.data.event_set import EventSet


def save_event_set(
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
    df = evset.to_dataframe()
    df.to_csv(path, index=False, sep=sep, na_rep=na_rep, columns=columns)
