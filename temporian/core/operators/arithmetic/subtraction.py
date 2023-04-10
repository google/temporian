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

"""Subtract one event from another"""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.arithmetic.base import BaseArithmeticOperator
from temporian.core.operators.arithmetic import Resolution


class SubtractionOperator(BaseArithmeticOperator):
    """
    Subtract second event from the first one
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "SUBTRACTION"

    @property
    def prefix(self) -> str:
        return "sub"

    def _get_output_dtype(self, input_dtype):
        return input_dtype


operator_lib.register_operator(SubtractionOperator)


def subtract(
    event_1: Event,
    event_2: Event,
    resolution: Resolution = Resolution.PER_FEATURE_IDX,
) -> Event:
    """
    Subtract two events.

    Args:
        event_1: First event
        event_2: Second event
        resolution: If resolution is Resolution.PER_FEATURE_IDX each feature
            will be subtract index wise. If resolution is
            Resolution.PER_FEATURE_NAME each feature of event_1 will be
            subtract with the feature in event_2 with the same name.
            Defaults to Resolution.PER_FEATURE_IDX.

    Returns:
        Event: Subtraction of event_1 and event_2 according to resolution.
    """
    return SubtractionOperator(
        event_1=event_1,
        event_2=event_2,
        resolution=resolution,
    ).outputs()["event"]
