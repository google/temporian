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

"""Interface for the Pandas implementations of the operators."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from temporal_feature_processor.implementation.pandas.data.event import \
    PandasEvent


class PandasOperator(ABC):
  """Base class to define an operator's interface."""

  @abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, PandasEvent]:
    """Apply the operator to its inputs.

    Returns:
        Dict[str, PandasEvent]: the output event of the operator.
    """

  def split_index(self, event: PandasEvent) -> Tuple[List[str], str]:
    """Split pandas' DataFrame index into  index columns and a timestamp column.

    Args:
        event (PandasEvent): input PandasEvent (pandas DataFrame).

    Returns:
        Tuple[List[str], str]: output index and timestamp names.
    """
    index_timestamp = event.index.names
    index = index_timestamp[:-1]
    timestamp = index_timestamp[-1]

    return index, timestamp
