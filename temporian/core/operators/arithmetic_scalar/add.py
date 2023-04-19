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

"""Arithmetic Addition Scalar Operator"""
from typing import Union

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class AddScalarOperator(BaseArithmeticScalarOperator):
    """
    Applies arithmetic addition between an event and a scalar value.
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "ADDITION_SCALAR"

    @property
    def prefix(self) -> str:
        return "add"


operator_lib.register_operator(AddScalarOperator)


def add_scalar(
    event: Event,
    value: Union[float, int, str, bool],
) -> Event:
    """
    Adds element-wise an event and a scalar value.

    Args:
        event: Event to perform addition to.
        value: Scalar value to add to all event features.

    Returns:
        Event: Event with the addition of event features and value.
    """
    return AddScalarOperator(
        event=event,
        value=value,
    ).outputs["event"]
