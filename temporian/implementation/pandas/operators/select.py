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

"""Implementation for the Select operator."""
from typing import Dict, List

from temporian.implementation.pandas.data.event import PandasEvent
from temporian.implementation.pandas.operators.base import PandasOperator


class PandasSelectOperator(PandasOperator):
    """Select a subset of features from an event."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    def __call__(self, input_event: PandasEvent) -> Dict[str, PandasEvent]:
        # get index
        event_index = input_event.index

        # get column positions
        column_positions = {
            col: pos for pos, col in enumerate(input_event.columns)
        }
        # get numpy array from DataFrame. This creates a view (doesn't
        # duplicate memory)
        input_event_arr = input_event.to_numpy()

        # select columns from array. Once again this creates a view of the
        # array
        input_event_arr_select = input_event_arr[
            :, [column_positions[column] for column in self.feature_names]
        ]
        # create PandasEvent from selected array. This creates a view of the
        # input array
        output_event = PandasEvent(
            input_event_arr_select,
            columns=self.feature_names,
            index=event_index,
        )
        return {"output_event": output_event}
