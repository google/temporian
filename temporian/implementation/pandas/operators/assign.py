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

"""Implementation for the Assign operator."""
from typing import Dict

from temporian.implementation.pandas.data.event import PandasEvent
from temporian.implementation.pandas.operators.base import PandasOperator
from temporian.implementation.pandas import utils


class PandasAssignOperator(PandasOperator):
    def __call__(
        self, left_event: PandasEvent, right_event: PandasEvent
    ) -> Dict[str, PandasEvent]:
        """Assign features to an event.

        Input event and features must have same index. Features cannot have more
        than one row for a single index + timestamp occurence. Output event will
        have same exact index and timestamps as input one. Assignment can be
        understood as a left join on the index and timestamp columns.

        Args:
            left_event (PandasEvent): event to assign the feature to.
            right_event (PandasEvent): features to assign to the event.

        Returns:
            PandasEvent: a new event with the features right.
        """
        # assert indexes are the same
        if left_event.index.names != right_event.index.names:
            raise IndexError("Assign sequences must have the same index names.")

        # get index column names
        index, timestamp = utils.get_index_and_timestamp_column_names(
            left_event
        )

        # check there's no repeated timestamps index-wise in the right sequence
        if index:
            max_timestamps = (
                right_event.reset_index()
                .groupby(index)[timestamp]
                .value_counts()
                .max()
            )
        else:
            max_timestamps = (
                right_event.reset_index()[timestamp].value_counts().max()
            )

        if max_timestamps > 1:
            raise ValueError(
                "Cannot have repeated timestamps in right EventSequence."
            )
        output_event = left_event.join(right_event, how="left", rsuffix="y")
        # make assignment
        return {"event": output_event}
