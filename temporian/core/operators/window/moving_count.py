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


class MovingCountOperator(BaseWindowOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "MOVING_COUNT"

    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        return DType.INT32


operator_lib.register_operator(MovingCountOperator)


def moving_count(
    input: Node,
    window_length: Duration,
    sampling: Optional[Node] = None,
) -> Node:
    """Computes the number of values in a sliding window over the node.

    For each t in sampling, and for each index and feature independently,
    returns at time t the number of non-nan values for the feature in the window
    (t - window_length, t].

    If `sampling` is provided samples the moving window's value at each
    timestamp in `sampling`, else samples it at each timestamp in `input`.

    If the window does not contain any values (e.g., all the values are missing,
    or the window does not contain any sampling), outputs missing values.

    Basic usage:
        ```python
        >>> a_evset = tp.event_set(
        ...     timestamps=[0, 1, 2, 5, 6, 7],
        ...     features={"value": [np.nan, 1, 5, 10, 15, 20]},
        ... )
        >>> a = a_evset.node()

        >>> result = tp.moving_count(a, tp.duration.seconds(2))
        >>> result.run({a: a_evset})
        indexes: ...
            (6 events):
                timestamps: [0. 1. 2. 5. 6. 7.]
                'value': [0 1 2 1 2 2]
        ...

        ```

    Example with external sampling:
        ```python
        >>> a_evset = tp.event_set(
        ...     timestamps=[0, 1, 2, 5],
        ...     features={"value": [np.nan, 1, 5, 10]},
        ... )
        >>> sampling_evset= tp.event_set(
        ...     timestamps=[-1, 0, 1, 2, 3, 4, 5, 6, 7],
        ... )
        >>> a = a_evset.node()
        >>> sampling = sampling_evset.node()

        >>> result = tp.moving_count(a, tp.duration.seconds(2), sampling)
        >>> result.run({a: a_evset, sampling: sampling_evset})
        indexes: ...
            (9 events):
                timestamps: [-1. 0. 1. 2. 3. 4. 5. 6. 7.]
                'value': [0 0 1 2 1 0 1 1 0]
        ...

        ```

    Example with indices:
        ```python
        >>> a_evset = tp.event_set(
        ...     timestamps=[1, 2, 3, 0, 1, 2],
        ...     features={
        ...         "value": [1, 1, 1, 1, 1, 1],
        ...         "idx": ["i1", "i1", "i1", "i2", "i2", "i2"],
        ...     },
        ...     indexes=["idx"],
        ... )
        >>> a = a_evset.node()
        >>> result = tp.moving_count(a, tp.duration.seconds(2))
        >>> result.run({a: a_evset})
        indexes: [('idx', str_)]
        features: [('value', int32)]
        events:
            idx=i2 (3 events):
                timestamps: [0. 1. 2.]
                'value': [1 2 2]
            idx=i1 (3 events):
                timestamps: [1. 2. 3.]
                'value': [1 2 2]
        ...

        ```

    Args:
        input: Node for which to count the number of values in each feature.
        window_length: Sliding window's length.
        sampling: Timestamps to sample the sliding window's value at. If not
            provided, timestamps in `input` are used.

    Returns:
        Node containing the non-nan count of each feature in `input`.
    """
    return MovingCountOperator(
        input=input,
        window_length=normalize_duration(window_length),
        sampling=sampling,
    ).outputs["output"]
