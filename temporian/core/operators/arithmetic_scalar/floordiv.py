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
from typing import Union, List

from temporian.core import operator_lib
from temporian.core.data import dtype as dtype_lib
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

    @property
    def supported_value_dtypes(self) -> List[dtype_lib.DType]:
        return [
            dtype_lib.FLOAT32,
            dtype_lib.FLOAT64,
            dtype_lib.INT32,
            dtype_lib.INT64,
        ]


operator_lib.register_operator(FloorDivScalarOperator)


SCALAR = Union[float, int]


def floordiv_scalar(
    numerator: Union[Event, SCALAR],
    denominator: Union[Event, SCALAR],
) -> Event:
    """
    Divides element-wise an event and a scalar and takes the result floor.

    Args:
        numerator: Numerator.
        denominator: Denominator.
    Returns:
        Event: Integer division of numerator features and denominator value
    """
    scalars_types = (float, int)

    if isinstance(numerator, Event) and isinstance(denominator, scalars_types):
        return FloorDivScalarOperator(
            event=numerator,
            value=denominator,
            is_value_first=False,
        ).outputs["event"]

    if isinstance(numerator, scalars_types) and isinstance(denominator, Event):
        return FloorDivScalarOperator(
            event=denominator,
            value=numerator,
            is_value_first=True,
        ).outputs["event"]

    raise ValueError(
        "Invalid input types for floordiv_scalar. "
        "Expected (Event, SCALAR) or (SCALAR, Event), "
        f"got ({type(numerator)}, {type(denominator)})."
    )
