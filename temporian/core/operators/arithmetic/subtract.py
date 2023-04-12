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


class SubtractOperator(BaseArithmeticOperator):
    """
    Subtract second event from the first one, feature to feature according
    to their position.
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "SUBTRACTION"

    @property
    def prefix(self) -> str:
        return "sub"


operator_lib.register_operator(SubtractOperator)


def subtract(
    event_1: Event,
    event_2: Event,
) -> Event:
    """
    Subtract event_2 from event_1.

    Args:
        event_1: First event
        event_2: Second event

    Returns:
        Event: Subtraction of event_2 features from event_1.
    """
    return SubtractOperator(
        event_1=event_1,
        event_2=event_2,
    ).outputs["event"]
