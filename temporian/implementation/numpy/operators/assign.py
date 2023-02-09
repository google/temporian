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


def _get_equal_sampling_indexes(
    sampling_1: "NumpySampling", sampling_2: "NumpySampling", index: tuple
) -> np.ndarray:
    """Get indexes of timestamps that are equal to sampling_1 & sampling_2
      in a specific sampling index.

    Args:
        sampling_1 (NumpySampling): sampling of assignee event.
        sampling_2 (NumpySampling): sampling of assigned event.
        index (tuple): index to check.

    Returns:
        np.ndarray: indexes of timestamps that are equal to sampling_1 & sampling_2.
    """
    return np.where(sampling_1.data[index] == sampling_2.data[index])[0]


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
        # check both keys are the same
        if assignee_event.sampling.names != assigned_event.sampling.names:
            raise ValueError("Assign sequences must have the same index names.")

        output = NumpyEvent(
            data=assignee_event.data.copy(), sampling=assignee_event.sampling
        )

        for index in assignee_event.data.keys():
            # If index is in assigned append the features to the output event
            if index in assigned_event.data.keys():
                # get indexes of timestamps that are equal to sampling_1 & sampling_2
                equal_sampling_indexes = _get_equal_sampling_indexes(
                    sampling_1=assignee_event.sampling,
                    sampling_2=assigned_event.sampling,
                    index=index,
                )
                for i, _ in enumerate(assigned_event.data[index]):
                    event_2_feature = assigned_event.data[index][i]
                    # filter indexes of timestamps that are equal to sampling_1 & sampling_2
                    event_2_feature.data = np.take(
                        event_2_feature.data, indices=equal_sampling_indexes
                    )
                    # change sampling to same as assignee
                    event_2_feature.sampling = assignee_event.sampling
                    output.data[index].append(event_2_feature)
            # If index is not in assigned, append the features of assigned with None values
            else:
                for feature_name in assigned_event.feature_names:
                    output.data[index].append(
                        NumpyFeature(
                            name=feature_name,
                            sampling=assignee_event.sampling,
                            # create a list of None values with the same length as assignee sampling
                            data=np.full(
                                len(assignee_event.sampling.data[index]), np.nan
                            ),
                        )
                    )

        # make assignment
        return {"event": output}
