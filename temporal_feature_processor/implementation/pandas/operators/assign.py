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

  def __call__(self, event: PandasEvent, features: PandasEvent) -> PandasEvent:
    """Assign features to an event.

    Input event and features must have same index. Features cannot have more
    than one row for a single index + timestamp occurence. Output event will
    have same exact index and timestamps as input one. Assignment can be
    understood as a left join on the index and timestamp columns.

    Args:
        event (PandasEvent): event to assign the feature to.
        feature (PandasEvent): features to assign to the event.

    Returns:
        PandasEvent: a new event with the features assigned.
    """

    raise NotImplementedError()
