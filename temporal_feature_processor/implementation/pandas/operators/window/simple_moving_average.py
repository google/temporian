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

from typing import Dict

import pandas as pd

from temporal_feature_processor.implementation.pandas.data.event import \
    PandasEvent
from temporal_feature_processor.implementation.pandas.data.sampling import \
    PandasSampling
from temporal_feature_processor.implementation.pandas.operators.window.base import \
    PandasWindowOperator


class PandasSimpleMovingAverageOperator(PandasWindowOperator):
  """Base class for window operators."""

  def __init__(self, window_length: str) -> None:
    super().__init__(window_length=window_length)

  def __call__(
      self,
      data: PandasEvent,
      sampling: PandasSampling,
  ) -> Dict[str, PandasEvent]:
    """Apply a simple moving average to an event.

    If input has more than one feature, the moving average will be computed for
    each of its features independently.

    Args:
      data: the input event to apply a simple moving average to.
      sampling: the desired sampling for the output event.

    Returns:
      Dict[str, PandasEvent]: the output event of the operator.
    """
    # remove index to be able to filter using index values
    data_no_index = data.reset_index()

    # get index columns and name of timestamp column
    index_columns = sampling.names[:-1]
    timestamp_column = sampling.names[-1]

    # create output event with desired sampling
    output = PandasEvent({
        "simple_moving_average": [None] * len(sampling)
    },
                         dtype=float).set_index(sampling)

    # manual rolling window since pandas doesn't support custom sampling in .rolling()
    # TODO: optimize window calculation
    for values in sampling:
      index_value = values[:-1]
      timestamp = values[-1]

      # filter by index_value
      data_filtered = data_no_index[(
          data_no_index[index_columns]
          == index_value).squeeze()] if index_columns else data_no_index

      # filter by window start/end dates
      data_filtered = data_filtered[
          (data_filtered[timestamp_column] <= timestamp) &
          (data_filtered[timestamp_column] >= timestamp -
           pd.Timedelta(self.window_length))]

      # calculate average of window
      mean = data_filtered.set_index(sampling.names).mean().item()

      # set result in output event
      loc = index_value + (timestamp,)
      output.loc[loc] = mean

    return {"output": output}
