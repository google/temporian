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

from abc import ABC, abstractmethod  # pylint: disable=g-multiple-import, g-importing-member
from typing import Any

from temporal_feature_processor.implementation.pandas.data.event import PandasEvent


class PandasOperator(ABC):
  """Base class to define an operator's interface."""

  @abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> PandasEvent:
    """Apply the operator to its inputs.

    Returns:
        PandasEvent: the output event of the operator.
    """
