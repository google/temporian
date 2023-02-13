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


def _get_common_timestamps(
    sampling_1: NumpySampling, sampling_2: NumpySampling, index: tuple
) -> np.ndarray:
    """It returns a np.ndarray with same shape as sampling_1.data[index] with True
    values where sampling_1 and sampling_2 have the same timestamp.

    Args:
        sampling_1 (NumpySampling): sampling of assignee event.
        sampling_2 (NumpySampling): sampling of assigned event.
        index (tuple): index to check.

    Returns:
        np.ndarray: boolean np.darray with same shape as sampling_1.data[index]

    Examples:
        >>> sampling_1 = NumpySampling(
        ...     names=["user_id"],
        ...     data={
        ...         (151591562,): np.array(
        ...             ["2022-02-05", "2022-02-06", "2022-02-07"],
        ...             dtype="datetime64",
        ...         ),
        ...         }
        ...     )
        >>> sampling_2 = NumpySampling(
        ...     names=["user_id"],
        ...     data={
        ...         (151591562,): np.array(
        ...             ["2022-02-06", "2022-02-07", "2022-02-08"],
        ...             dtype="datetime64",
        ...         ),
        ...         }
        ...     )
        >>> _get_common_timestamps(sampling_1, sampling_2, (151591562,))
        array([False, True, True])

    """
    return np.in1d(sampling_1.data[index], sampling_2.data[index])


def _convert_assigned_feature_to_assignee_sampling(
    feature: NumpyFeature,
    common_timestamps: List[bool],
    assignee_sampling: NumpySampling,
    assigned_sampling: NumpySampling,
    index: tuple,
) -> NumpyFeature:
    """Converts assigned feature to assignee sampling. It requires a boolean list with
    same length as the assignee sampling, where True values indicate that the
    assigned feature has the same timestamp.

    Args:
        feature (NumpyFeature): assigned feature to convert.
        common_timestamps (List[bool]): list of booleans indicating where feature's sampling has same timestamp as new sampling
        assignee_sampling (NumpySampling): sampling of assignee event.
        assigned_sampling (NumpySampling): sampling of assigned event.
        index (tuple): index to check.

    Returns:
        NumpyFeature: feature with new sampling.

    """
    new_feature = NumpyFeature(
        name=feature.name,
        data=np.full(len(common_timestamps), np.nan),
    )

    # Loop over common timestamps and if True, then add the assigned feature value
    # in the common timestamp.
    # Assuming that the feature data is sorted by timestamp, we can use binary search
    # to find the index of the common timestamp in the feature data, instead of
    # numpy where search. This would make the algorithm O(n * (1 + log(m)))
    # (n: len(common_timestamps), m: len(feature.data). Adding extra n for
    # np.full at the beginning.
    # We can make this more efficient by trimming the feature data to the
    # last found array index because as it sorted by timestamps, the next one
    # will be after that one.

    for i, is_common in enumerate(common_timestamps):
        last_common_index = 0
        if is_common:
            # Get the sampling index of the common timestamp
            common_timestamp = assignee_sampling.data[index][i]
            # Get the index of the common timestamp in the feature
            # uses binary search because feature.data is sorted by timestamp
            # last common index is used to speed up the search by trimming the
            # feature data to the last found array index because as it sorted
            # by timestamps, the next one will be after that one.
            common_index = np.searchsorted(
                assigned_sampling.data[index][last_common_index:],
                common_timestamp,
            )
            # Assign the feature value to the new feature
            new_feature.data[i] = feature.data[common_index]
            last_common_index = common_index

    return new_feature


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
            # If index is in assigned append the features to the output event
            if index in assigned_event.data.keys():
                # Get timestamps that are equal to sampling_1 & sampling_2
                common_timestamps = _get_common_timestamps(
                    sampling_1=assignee_event.sampling,
                    sampling_2=assigned_event.sampling,
                    index=index,
                )
                # Loop over assigned features
                for assigned_feature in assigned_event.data[index]:
                    # convert feature to new sampling
                    assigned_feature_filtered = (
                        _convert_assigned_feature_to_assignee_sampling(
                            feature=assigned_feature,
                            common_timestamps=common_timestamps,
                            assignee_sampling=assignee_event.sampling,
                            assigned_sampling=assigned_event.sampling,
                            index=index,
                        )
                    )
                    output.data[index].append(assigned_feature_filtered)
            # If index is not in assigned, append the features of assigned with None values
            else:
                for feature_name in assigned_event.feature_names:
                    output.data[index].append(
                        NumpyFeature(
                            name=feature_name,
                            # Create a list of None values with the same length as assignee sampling
                            data=np.full(
                                len(assignee_event.sampling.data[index]), np.nan
                            ),
                        )
                    )

        # make assignment
        return {"event": output}
