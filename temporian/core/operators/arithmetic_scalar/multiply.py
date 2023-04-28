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

"""Multiply scalar operator class and public API function definition."""

from typing import Union, List

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.event import Event
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class MultiplyScalarOperator(BaseArithmeticScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "MULTIPLICATION_SCALAR"

    @property
    def supported_value_dtypes(self) -> List[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]


operator_lib.register_operator(MultiplyScalarOperator)


def multiply_scalar(
    event: Event,
    value: Union[float, int],
) -> Event:
    """Multiplies an event by a scalar value.

    Each item in each feature in `event` is multiplied by `value`.

    Args:
        event: Event to multiply.
        value: Scalar value to multiply the event by.

    Returns:
        Event with the multiplication of event features and value.
    """
    return MultiplyScalarOperator(
        event=event,
        value=value,
    ).outputs["event"]
