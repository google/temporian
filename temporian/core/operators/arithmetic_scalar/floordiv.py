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

"""Floor or integer division scalar Operator"""
from typing import Union

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class FloorDivScalarOperator(BaseArithmeticScalarOperator):
    """
    Integer division of an event and a scalar (i.e: a//10)
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "FLOORDIV_SCALAR"

    @property
    def prefix(self) -> str:
        return "floordiv"


operator_lib.register_operator(FloorDivScalarOperator)


def floordiv_scalar(
    numerator: Event,
    denominator: Union[float, int, str, bool],
) -> Event:
    """
    Divides element-wise an event and a scalar and takes the result floor.

    Args:
        numerator: Numerator event
        denominator: Denominator scalar value
    Returns:
        Event: Integer division of numerator features and denominator value
    """
    return FloorDivScalarOperator(
        event=numerator,
        value=denominator,
    ).outputs["event"]
