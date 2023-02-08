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

import numpy as np
from temporian.implementation.numpy.data.event import NumpyEvent


class NumpyAssignOperator:
    def __call__(
        self, assignee_event: NumpyEvent, assigned_event: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        """Assign features to an event.

        Input event and features must have same index. Features cannot have more
        than one row for a single index + timestamp occurence. Output event will
        have same exact index and timestamps as input one. Assignment can be
        understood as a left join on the index and timestamp columns.

        Args:
            assignee_event (NumpyEvent): event to assign the feature to.
            assigned_event (NumpyEvent): features to assign to the event.

        Returns:
            NumpyEvent: a new event with the features assigned.
        """
        output = NumpyEvent(data=assignee_event.data.copy(), sampling=assignee_event.sampling)

        for index in assignee_event.data.keys():
            if index in assigned_event.data.keys():
                different_sampling = np.where(np.array(assignee_event.sampling.data[index]) != np.array(assigned_event.sampling.data[index]))[0]
                for i, _ in enumerate(assigned_event.data[index]):
                    event_2_feature = assigned_event.data[index][i]
                    # get values with same timestamp as event 1
                    event_2_feature = (event_2_feature[0], np.delete(event_2_feature[1], different_sampling))
                    output.data[index].append(event_2_feature)
            else:
                for i, _ in enumerate(event_1[index]):
                    output.data[index][i] += (None, [None for _ in range(len(sampling_1[index]))])

        # make assignment
        return {"event": output}
