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

"""Floor or integer division Operator"""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.arithmetic.base import BaseArithmeticOperator


class FloorDivOperator(BaseArithmeticOperator):
    """
    Integer division of first event by second one (i.e: a//b)
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "FLOORDIV"

    @property
    def prefix(self) -> str:
        return "floordiv"


operator_lib.register_operator(FloorDivOperator)


def floordiv(
    numerator: Event,
    denominator: Event,
) -> Event:
    """
    Divide two events and take the result floor.

    Args:
        numerator: Numerator event
        denominator: Denominator event
    Returns:
        Event: Integer division of numerator features and denominator features
    """
    return FloorDivOperator(
        event_1=numerator,
        event_2=denominator,
    ).outputs["event"]
