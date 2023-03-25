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

from typing import Dict, Optional
import numpy as np
import copy

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.core.operators.assign import AssignOperator
from temporian.implementation.numpy import implementation_lib


class AssignNumpyImplementation:
    def __init__(self, operator: AssignOperator):
        assert isinstance(operator, AssignOperator)
        self._operator = operator

    def __call__(
        self,
        event_1: NumpyEvent,
        event_2: NumpyEvent,
        event_3: Optional[NumpyEvent] = None,
        event_4: Optional[NumpyEvent] = None,
    ) -> Dict[str, NumpyEvent]:
        """Assign features to an event.

        Left and right must have same index names. right cannot have more
        than one row for a single index + timestamp occurence. Output event will
        have same exact index and timestamps (sampling) as the left event.

        Assignment is done by matching the timestamps and index of the left and right.
        The right features will be appended to the left features in the matching
        indexes. Index values in the right event missing in the left event will
        be filled with NaNs.  Index values present in the right event missing in
        the left event will be ignored.

        Args:
            left_event: event where features will be right to.
            right_event: event with features to assign.

        Returns:
            NumpyEvent: a new event with the features right.

        Raises:
            ValueError: if left and right events have different indexes names.
            ValueError: if right event has repeated timestamps for same index.

        """
        events = [event_1, event_2]
        if event_3 is not None:
            events.append(event_3)
        if event_4 is not None:
            events.append(event_4)

        dst_event = NumpyEvent(data={}, sampling=event_1.sampling)

        for index in event_1.data.keys():
            dst_features = []
            for event in events:
                dst_features.extend(copy.copy(event.data[index]))
            dst_event.data[index] = dst_features

        # make assignment
        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    AssignOperator, AssignNumpyImplementation
)
