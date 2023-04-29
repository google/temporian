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

"""Multiply operator class and public API function definition."""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.arithmetic.base import BaseArithmeticOperator


class MultiplyOperator(BaseArithmeticOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "MULTIPLICATION"

    @property
    def prefix(self) -> str:
        return "mult"


operator_lib.register_operator(MultiplyOperator)


def multiply(
    event_1: Event,
    event_2: Event,
) -> Event:
    """Multiplies two events.

    Each feature in `event_1` is multiplied by the feature in `event_2` in the
    same position.

    `event_1` and `event_2` must have the same sampling and the same number of
    features.

    Args:
        event_1: First event.
        event_2: Second event.

    Returns:
        Multiplication of `event_1`'s and `event_2`'s features.
    """
    return MultiplyOperator(
        event_1=event_1,
        event_2=event_2,
    ).outputs["event"]
