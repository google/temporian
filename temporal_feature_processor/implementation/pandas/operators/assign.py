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

from temporal_feature_processor.implementation.pandas.data.event import PandasEvent
from temporal_feature_processor.implementation.pandas.operators.base import PandasOperator


class PandasAssignOperator(PandasOperator):

  def __call__(
      self, assignee_event: PandasEvent, assigned_event: PandasEvent
  ) -> PandasEvent:
    """Assign features to an event.

    Input event and features must have same index. Features cannot have more
    than one row for a single index + timestamp occurence. Output event will
    have same exact index and timestamps as input one. Assignment can be
    understood as a left join on the index and timestamp columns.

    Args:
        assignee_event (PandasEvent): event to assign the feature to.
        assigned_event (PandasEvent): features to assign to the event.

    Returns:
        PandasEvent: a new event with the features assigned.
    """
    # assert indexes are the same
    if assignee_event.index.names != assigned_event.index.names:
      raise IndexError("Assign sequences must have the same index names.")

    # get index column names
    index, timestamp = self.split_index(assignee_event)

    # check there's no repeated timestamps index-wise in the assigned sequence
    if index:
      max_timestamps = (
          assigned_event.reset_index()
          .groupby(index)[timestamp]
          .value_counts()
          .max()
      )
    else:
      max_timestamps = (
          assigned_event.reset_index()[timestamp].value_counts().max()
      )

    if max_timestamps > 1:
      raise ValueError(
          "Cannot have repeated timestamps in assigned EventSequence."
      )

    # make assignment
    output = assignee_event.join(assigned_event, how="left", rsuffix="y")
    return {"output": output}
