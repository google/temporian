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

"""Implementation for the SimpleMovingAverage operator."""

from temporal_feature_processor.implementation.pandas.data.event import PandasEvent
from temporal_feature_processor.implementation.pandas.data.sampling import PandasSampling
from temporal_feature_processor.implementation.pandas.operators.window.base import PandasWindowOperator


class PandasSimpleMovingAverageOperator(PandasWindowOperator):
  """Base class for window operators."""

  def __init__(self, window_length: int) -> None:
    super().__init__(window_length=window_length)

  def __call__(
      self,
      data: PandasEvent,
      sampling: PandasSampling,
  ) -> PandasEvent:
    """Apply a simple moving average to an event.

    If input has more than one feature, the moving average will be computed for
    each of its features independently.

    Args:
        data (PandasEvent): the input event to apply a simple moving average to.

    Returns:
        PandasEvent: the output of the operator.
    """

    raise NotImplementedError()
