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

"""Utility functions for the Pandas implementation."""
from typing import Union
from temporian.implementation.pandas.data import event
from temporian.implementation.pandas.data import sampling


def get_index_and_timestamp_column_names(
    obj: Union[event.PandasEvent, sampling.PandasSampling]
) -> tuple[list[str], str]:
    """Get the names of the index columns and the timestamp column from a
  PandasEvent or PandasSampling.

  Args:
      obj (Union[event.PandasEvent, sampling.PandasSampling]): the event or
      sampling to get the index names of.

  Returns:
      Tuple[List[str], str]: output index and timestamp column names.
  """
    if isinstance(obj, event.PandasEvent):
        obj = obj.index

    index_cols = obj.names[:-1]
    timestamp_col = obj.names[-1]

    return index_cols, timestamp_col
