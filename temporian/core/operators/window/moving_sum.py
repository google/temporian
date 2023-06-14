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

"""Moving Sum operator class and public API function definition.."""

from typing import Optional

import numpy as np

from temporian.core import operator_lib
from temporian.core.data.duration import Duration, normalize_duration
from temporian.core.data.node import Node
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.window.base import BaseWindowOperator
from temporian.core.data.dtype import DType


class MovingSumOperator(BaseWindowOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "MOVING_SUM"

    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        return feature.dtype


operator_lib.register_operator(MovingSumOperator)


def moving_sum(
    input: Node,
    window_length: Duration,
    sampling: Optional[Node] = None,
) -> Node:
    """Computes the sum of values in a sliding window over the node.

    For each t in sampling, and for each feature independently, returns at time
    t the sum of the feature in the window [t - window_length, t].

    If `sampling` is provided samples the moving window's value at each
    timestamp in `sampling`, else samples it at each timestamp in `input`.

    Missing values (such as NaNs) are ignored.

    If the window does not contain any values (e.g., all the values are missing,
    or the window does not contain any sampling), outputs missing values.

    Args:
        input: Features to sum.
        window_length: Sliding window's length.
        sampling: Timestamps to sample the sliding window's value at. If not
            provided, timestamps in `input` are used.

    Returns:
        Node containing the moving sum of each feature in `input`.
    """
    return MovingSumOperator(
        input=input,
        window_length=normalize_duration(window_length),
        sampling=sampling,
    ).outputs["output"]


def cumsum(
    input: Node,
) -> Node:
    """Cumulative Sum.

    Foreach timestamp, calculate the sum of the feature from the beginning.
    It's a shorthand for `moving_sum(event, window_length=np.inf)`.

    Missing values are ignored.

    While the feature does not have any values (e.g., missing initial values),
    outputs missing values.

    Args:
        input: The node with features to accumulate.

    Returns:
        A node containing the cumulative sum of each feature in `node`.
    """
    return MovingSumOperator(
        input=input,
        window_length=normalize_duration(np.inf),
    ).outputs["output"]
