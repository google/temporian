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
        self, left_event: NumpyEvent, right_event: NumpyEvent
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
        if left_event.sampling.names != right_event.sampling.names:
            raise ValueError("Assign sequences must have the same index names.")

        if right_event.sampling.has_repeated_timestamps:
            raise ValueError(
                "right sequence cannot have repeated timestamps in the same"
                " index."
            )

        output = NumpyEvent(
            data=left_event.data.copy(), sampling=left_event.sampling
        )

        for index in left_event.data.keys():
            output_data = output.data[index]
            left_sampling_data = left_event.sampling.data[index]
            number_timestamps = len(left_sampling_data)

            # If index is in right append the features to the output event
            if index in right_event.data:
                right_sampling_data = right_event.sampling.data[index]

                mask = (
                    left_sampling_data[:, np.newaxis]
                    == right_sampling_data[np.newaxis, :]
                )
                mask_i, mask_j = mask.nonzero()

                for feature in right_event.data[index]:
                    new_feature = NumpyFeature(
                        name=feature.name,
                        data=np.full(number_timestamps, np.nan),
                    )
                    new_feature.data[mask_i] = feature.data[mask_j]
                    output_data.append(new_feature)

            # If index is not in right, append the features of right with None values
            else:
                for feature_name in right_event.feature_names:
                    output_data.append(
                        NumpyFeature(
                            name=feature_name,
                            # Create a list of None values with the same length as left sampling
                            data=np.full(number_timestamps, np.nan),
                        )
                    )

        # make assignment
        return {"event": output}
