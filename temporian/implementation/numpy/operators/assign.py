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
from typing import Dict, List

import numpy as np
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


class NumpyAssignOperator:
    def __call__(
        self, assignee_event: NumpyEvent, assigned_event: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        """Assign features to an event.

        Assignee and assigned must have same index names. Assigned cannot have more
        than one row for a single index + timestamp occurence. Output event will
        have same exact index and timestamps (sampling) as the assignee event.

        Assignment is done by matching the timestamps and index of the assignee and assigned.
        The assigned features will be appended to the assignee features in the matching
        indexes. Index values in the assigned event missing in the assignee event will
        be filled with NaNs.  Index values present in the assigned event missing in
        the assignee event will be ignored.

        Args:
            assignee_event (NumpyEvent): event where features will be assigned to.
            assigned_event (NumpyEvent): event with features to assign.

        Returns:
            NumpyEvent: a new event with the features assigned.

        Raises:
            ValueError: if assignee and assigned events have different indexes names.
            ValueError: if assigned event has repeated timestamps for same index.

        """
        if assignee_event.sampling.names != assigned_event.sampling.names:
            raise ValueError("Assign sequences must have the same index names.")

        if assigned_event.sampling.has_repeated_timestamps:
            raise ValueError(
                "Assigned sequence cannot have repeated timestamps in the same"
                " index."
            )

        output = NumpyEvent(
            data=assignee_event.data.copy(), sampling=assignee_event.sampling
        )

        for index in assignee_event.data.keys():
            output_data = output.data[index]
            assignee_sampling_data = assignee_event.sampling.data[index]
            number_timestamps = len(assignee_sampling_data)

            # If index is in assigned append the features to the output event
            if index in assigned_event.data:
                assigned_sampling_data = assigned_event.sampling.data[index]

                mask = (
                    assignee_sampling_data[:, np.newaxis]
                    == assigned_sampling_data[np.newaxis, :]
                )
                mask_i, mask_j = mask.nonzero()

                for feature in assigned_event.data[index]:
                    new_feature = NumpyFeature(
                        name=feature.name,
                        data=np.full(number_timestamps, np.nan),
                    )
                    new_feature.data[mask_i] = feature.data[mask_j]
                    output_data.append(new_feature)

            # If index is not in assigned, append the features of assigned with None values
            else:
                for feature_name in assigned_event.feature_names:
                    output_data.append(
                        NumpyFeature(
                            name=feature_name,
                            # Create a list of None values with the same length as assignee sampling
                            data=np.full(number_timestamps, np.nan),
                        )
                    )

        # make assignment
        return {"event": output}
