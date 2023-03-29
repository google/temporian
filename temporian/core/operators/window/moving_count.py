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

"""Moving Count operator."""
from typing import Optional, List

from temporian.core import operator_lib
from temporian.core.data.dtype import INT64
from temporian.core.data.duration import Duration
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.window.base import BaseWindowOperator


class MovingCountOperator(BaseWindowOperator):
    """
    Window operator to compute the moving count.
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "MOVING_COUNT"

    @property
    def prefix(self) -> str:
        return "moving_count"

    def output_features(
        self,
        event: Event,
        sampling: Sampling,
    ) -> List[Feature]:
        features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"{self.prefix}_{f.name()}",
                dtype=INT64,
                sampling=sampling,
                creator=self,
            )
            for f in event.features()
        ]

        return features


operator_lib.register_operator(MovingCountOperator)


def moving_count(
    event: Event,
    window_length: Duration,
    sampling: Optional[Event] = None,
) -> Event:
    """Moving Count.

    For each sampling, and for each feature independently, returns at time "t"
    the number of non values of the feature in the time windows [t-window, t].

    If "sampling" is provided, applies the operator for each timestamps of
    "sampling". If "sampling" is not provided, apply the operator for each
    timestamps of "event".

    Missing values are ignored.

    If the window does not contain any values (e.g., all the values are missing,
    or the window does not contain any sampling), outputs missing values.

    Args:
        event: The features to count.
        window_length: The window length for counting.
        sampling: If provided, define when the operator is applied. If not
          provided, the operator is applied for each timestamp of "event".

    Returns:
        An event containing the moving count of each feature in "event".
    """
    return MovingCountOperator(
        event=event,
        window_length=window_length,
        sampling=sampling,
    ).outputs()["event"]
