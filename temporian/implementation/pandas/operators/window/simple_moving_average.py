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

from typing import Dict, Optional

import pandas as pd

from temporian.implementation.pandas.data.event import PandasEvent
from temporian.implementation.pandas.data.sampling import PandasSampling
from temporian.implementation.pandas.operators.window.base import (
    PandasWindowOperator,
)
from temporian.implementation.pandas import utils


class PandasSimpleMovingAverageOperator(PandasWindowOperator):
    """Pandas implementation for the simple moving average operator."""

    def __init__(self, window_length: str) -> None:
        super().__init__(window_length=window_length)

    def __call__(
        self,
        data: PandasEvent,
        sampling: Optional[PandasEvent] = None,
    ) -> Dict[str, PandasEvent]:
        """Apply a simple moving average to an event.

        If input has more than one feature, the moving average will be computed for
        each of its features independently.

        Args:
          data: the input event to apply a simple moving average to.
          sampling: an event with the desired sampling for the output event.
              If None, the original sampling of `data` will be used.

        Returns:
          Dict[str, PandasEvent]: the output event of the operator.
        """
        # remove index to be able to filter using index values
        data_no_index = data.reset_index()

        if sampling is None:
            sampling = data

        # get index columns and name of timestamp column
        (
            index_columns,
            timestamp_column,
        ) = utils.get_index_and_timestamp_column_names(sampling)

        # create output event with desired sampling
        output = PandasEvent(
            {f"sma_{col}": [None] * len(sampling) for col in data.columns},
            dtype=float,
        ).set_index(sampling.index)

        # manual rolling window since pandas doesn't support custom sampling in .rolling()
        # TODO: optimize window calculation
        for values in sampling.index:
            index_value = values[:-1]
            timestamp = values[-1]

            # filter by index_value
            data_filtered = (
                data_no_index[
                    (data_no_index[index_columns] == index_value).squeeze()
                ]
                if index_columns
                else data_no_index
            )

            # filter by window start/end dates
            data_filtered = data_filtered[
                (data_filtered[timestamp_column] <= timestamp)
                & (
                    data_filtered[timestamp_column]
                    >= timestamp - pd.Timedelta(self.window_length)
                )
            ]

            # calculate average of window
            mean = (
                data_filtered.set_index(sampling.index.names)
                .mean(axis=0)
                .values
            )

            # set result in output event
            loc = index_value + (timestamp,)
            output.loc[loc] = mean

        return {"output": output}
