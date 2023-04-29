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

"""Simple Moving Average operator."""
from typing import Optional, List

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.duration import Duration
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.window.base import BaseWindowOperator


class SimpleMovingAverageOperator(BaseWindowOperator):
    """
    Window operator to compute the simple moving average.
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "SIMPLE_MOVING_AVERAGE"

    def get_feature_dtype(self, feature: Feature) -> str:
        return (
            DType.FLOAT32 if feature.dtype == DType.FLOAT32 else DType.FLOAT64
        )


operator_lib.register_operator(SimpleMovingAverageOperator)


def simple_moving_average(
    event: Event,
    window_length: Duration,
    sampling: Optional[Event] = None,
) -> Event:
    """Computes the average of values in a sliding window over the event.

    For each t in sampling, and for each feature independently, returns at time
    t the average value of the feature in the window [t - window_length, t].

    If `sampling` is provided samples the moving window's value at each
    timestamp in `sampling`, else samples it at each timestamp in `event`.

    Missing values (such as NaNs) are ignored.

    If the window does not contain any values (e.g., all the values are missing,
    or the window does not contain any sampling), outputs missing values.

    Args:
        event: Features to average.
        window_length: Sliding window's length.
        sampling: Timestamps to sample the sliding window's value at. If not
            provided, timestamps in `event` are used.

    Returns:
        Event containing the moving average of each feature in `event`.
    """
    return SimpleMovingAverageOperator(
        event=event,
        window_length=window_length,
        sampling=sampling,
    ).outputs["event"]
