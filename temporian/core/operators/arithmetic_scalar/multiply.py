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

"""Arithmetic Multiplication Scalar Operator"""
from typing import Union

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class MultiplyScalarOperator(BaseArithmeticScalarOperator):
    """
    Applies arithmetic multiplication to an event and a scalar value.
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "MULTIPLICATION_SCALAR"

    @property
    def prefix(self) -> str:
        return "mult"


operator_lib.register_operator(MultiplyScalarOperator)


def multiply_scalar(
    event: Event,
    value: Union[float, int, str, bool],
) -> Event:
    """
    Multiplies element-wise an event features and a scalar value.

    Args:
        event_1: event to perform multiplication to.
        value: scalar value to multiply to all event features.

    Returns:
        Event: event with the multiplication of event features and value.
    """
    return MultiplyScalarOperator(
        event=event,
        value=value,
    ).outputs["event"]
