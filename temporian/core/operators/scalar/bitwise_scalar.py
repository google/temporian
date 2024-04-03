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

"""Event/scalar arithmetic operators classes and public API definitions."""

from typing import Union

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import EventSetNode
from temporian.core.operators.scalar.base import (
    BaseScalarOperator,
)
from temporian.core.typing import EventSetOrNode


class BitwiseAndScalarOperator(BaseScalarOperator):
    DEF_KEY = "BITWISE_AND_SCALAR"

class BitwiseOrScalarOperator(BaseScalarOperator):
    DEF_KEY = "BITWISE_OR_SCALAR"

class BitwiseXorScalarOperator(BaseScalarOperator):
    DEF_KEY = "BITWISE_XOR_SCALAR"


@compile
def bitwise_and_scalar(
    input: EventSetOrNode,
    value: int,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return BitwiseAndScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


@compile
def bitwise_or_scalar(
    input: EventSetOrNode,
    value: int,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return BitwiseOrScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


@compile
def bitwise_xor_scalar(
    input: EventSetOrNode,
    value: int,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return BitwiseXorScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


operator_lib.register_operator(BitwiseAndScalarOperator)
operator_lib.register_operator(BitwiseOrScalarOperator)
operator_lib.register_operator(BitwiseXorScalarOperator)
