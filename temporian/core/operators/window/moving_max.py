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

"""Moving count operator class and public API function definition."""

from typing import Optional

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.duration_utils import Duration, normalize_duration
from temporian.core.data.node import Node
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.window.base import BaseWindowOperator


class MovingMaxOperator(BaseWindowOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "MOVING_MAX"

    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        return feature.dtype


operator_lib.register_operator(MovingMaxOperator)


def moving_max(
    input: Node,
    window_length: Duration,
    sampling: Optional[Node] = None,
) -> Node:
    """Computes the maximum in a sliding window over the node.

    For each t in sampling, and for each index and feature independently,
    returns at time t the max of non-nan values for the feature in the window
    (t - window_length, t].

    If `sampling` is provided samples the moving window's value at each
    timestamp in `sampling`, else samples it at each timestamp in `input`.

    If the window does not contain any values (e.g., all the values are missing,
    or the window does not contain any sampling), outputs missing values.

    Basic usage:
        ```python
        >>> a_evset = tp.event_set(
        ...     timestamps=[0, 1, 2, 5, 6, 7],
        ...     features={"value": [np.nan, 1, 5, 1, 15, 20]},
        ... )
        >>> a = a_evset.node()

        >>> result = tp.moving_max(a, tp.duration.seconds(4))
        >>> result.run({a: a_evset})
        indexes: ...
            (6 events):
                timestamps: [0. 1. 2. 5. 6. 7.]
                'value': [nan 1. 5. 5. 15. 20.]
        ...

        ```

    See [`tp.moving_count()`](../moving_count) for examples with external
    sampling and indices.

    Args:
        input: Node for which to count the number of values in each feature.
        window_length: Sliding window's length.
        sampling: Timestamps to sample the sliding window's value at. If not
            provided, timestamps in `input` are used.

    Returns:
        Node containing the max of each feature in `input`.
    """
    return MovingMaxOperator(
        input=input,
        window_length=normalize_duration(window_length),
        sampling=sampling,
    ).outputs["output"]
