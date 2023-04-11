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

"""Arithmetic Multiplication Operator"""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.arithmetic.base import BaseArithmeticOperator


class MultiplicationOperator(BaseArithmeticOperator):
    """
    Apply arithmetic multiplication to two events, feature to feature according
    to their position.
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "MULTIPLICATION"

    @property
    def prefix(self) -> str:
        return "mult"


operator_lib.register_operator(MultiplicationOperator)


def multiply(
    event_1: Event,
    event_2: Event,
) -> Event:
    """
    Multiply two events, according to feature position.

    Args:
        event_1: First event
        event_2: Second event

    Returns:
        Event: Multiplication of event_1 features and event_2 features.
    """
    return MultiplicationOperator(
        event_1=event_1,
        event_2=event_2,
    ).outputs()["event"]
