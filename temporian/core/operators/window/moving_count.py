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
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.duration_utils import Duration, normalize_duration
from temporian.core.data.node import EventSetNode
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.window.base import BaseWindowOperator
from temporian.core.typing import EventSetOrNode


class MovingCountOperator(BaseWindowOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "MOVING_COUNT"

    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        return DType.INT32

    def feature_schema(self, input: EventSetNode):
        return [FeatureSchema(name="count", dtype=DType.INT32)]


operator_lib.register_operator(MovingCountOperator)


@compile
def moving_count(
    input: EventSetOrNode,
    window_length: Duration,
    sampling: Optional[EventSetOrNode] = None,
) -> EventSetOrNode:
    """Computes the number of samplings in a sliding window over an
    [`EventSet`][temporian.EventSet].

    For each t in sampling, and for each index and feature independently,
    returns at time t the number of values in the window (t - window_length, t].

    Ignores input features. Always return a single feature called "count" of
    type tp.int32.

    If `sampling` is provided samples the moving window's value at each
    timestamp in `sampling`, else samples it at each timestamp in `input`.

    Basic example:
        ```python
        >>> a = tp.event_set(timestamps=[0, 1, 2, 5, 6, 7])
        >>> b = tp.moving_count(a, tp.duration.seconds(2))
        >>> b
        indexes: ...
            (6 events):
                timestamps: [0. 1. 2. 5. 6. 7.]
                'count': [1 2 2 1 2 2]
        ...

        ```

    Example with external sampling:
        ```python
        >>> a = tp.event_set(timestamps=[0, 1, 2, 5])
        >>> b = tp.event_set(timestamps=[-1, 0, 1, 2, 3, 4, 5, 6, 7])
        >>> c = tp.moving_count(a, tp.duration.seconds(2), sampling=b)
        >>> c
        indexes: ...
            (9 events):
                timestamps: [-1. 0. 1. 2. 3. 4. 5. 6. 7.]
                'count': [0 1 2 2 1 0 1 1 0]
        ...

        ```

    Example with index:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3, 0, 1, 2],
        ...     features={
        ...         "idx": ["i1", "i1", "i1", "i2", "i2", "i2"],
        ...     },
        ...     indexes=["idx"],
        ... )
        >>> b = tp.moving_count(a, tp.duration.seconds(2))
        >>> b
        indexes: [('idx', str_)]
        features: [('count', int32)]
        events:
            idx=b'i1' (3 events):
                timestamps: [1. 2. 3.]
                'count': [1 2 2]
            idx=b'i2' (3 events):
                timestamps: [0. 1. 2.]
                'count': [1 2 2]
        ...

        ```

    Args:
        input: EventSet for which to count the number of values in each feature.
        window_length: Sliding window's length.
        sampling: Timestamps to sample the sliding window's value at. If not
            provided, timestamps in `input` are used.

    Returns:
        EventSet containing the non-nan count of each feature in `input`.
    """
    assert isinstance(input, EventSetNode)
    if sampling is not None:
        assert isinstance(sampling, EventSetNode)

    return MovingCountOperator(
        input=input,
        window_length=normalize_duration(window_length),
        sampling=sampling,
    ).outputs["output"]
