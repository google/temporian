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

"""Binary arithmetic operators classes and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import EventSetNode
from temporian.core.data.dtype import DType
from temporian.core.operators.binary.base import BaseBinaryOperator
from temporian.core.typing import EventSetOrNode


class BaseArithmeticOperator(BaseBinaryOperator):
    DEF_KEY = ""
    PREFIX = ""

    @classmethod
    def operator_def_key(cls) -> str:
        return cls.DEF_KEY

    @property
    def prefix(self) -> str:
        return self.PREFIX


class AddOperator(BaseArithmeticOperator):
    DEF_KEY = "ADDITION"
    PREFIX = "add"


class SubtractOperator(BaseArithmeticOperator):
    DEF_KEY = "SUBTRACTION"
    PREFIX = "sub"


class MultiplyOperator(BaseArithmeticOperator):
    DEF_KEY = "MULTIPLICATION"
    PREFIX = "mult"


class FloorDivOperator(BaseArithmeticOperator):
    DEF_KEY = "FLOORDIV"
    PREFIX = "floordiv"


class ModuloOperator(BaseArithmeticOperator):
    DEF_KEY = "MODULO"
    PREFIX = "mod"


class PowerOperator(BaseArithmeticOperator):
    DEF_KEY = "POWER"
    PREFIX = "pow"


class DivideOperator(BaseArithmeticOperator):
    DEF_KEY = "DIVISION"
    PREFIX = "div"

    def __init__(
        self,
        input_1: EventSetNode,
        input_2: EventSetNode,
    ):
        super().__init__(input_1, input_2)

        # Assuming previous dtype check of input_1 and input_2 features
        for feat in input_1.schema.features:
            if feat.dtype in [DType.INT32, DType.INT64]:
                raise ValueError(
                    "Cannot use the divide operator on feature "
                    f"{feat.name} of type {feat.dtype}. Cast to "
                    "a floating point type or use "
                    "floordiv operator (//) instead, on these integer types."
                )


@compile
def add(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_1, EventSetNode)
    assert isinstance(input_2, EventSetNode)

    return AddOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def subtract(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_1, EventSetNode)
    assert isinstance(input_2, EventSetNode)

    return SubtractOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def multiply(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_1, EventSetNode)
    assert isinstance(input_2, EventSetNode)

    return MultiplyOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def divide(
    numerator: EventSetOrNode,
    denominator: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(numerator, EventSetNode)
    assert isinstance(denominator, EventSetNode)

    return DivideOperator(
        input_1=numerator,
        input_2=denominator,
    ).outputs["output"]


@compile
def floordiv(
    numerator: EventSetOrNode,
    denominator: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(numerator, EventSetNode)
    assert isinstance(denominator, EventSetNode)

    return FloorDivOperator(
        input_1=numerator,
        input_2=denominator,
    ).outputs["output"]


@compile
def modulo(
    numerator: EventSetOrNode,
    denominator: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(numerator, EventSetNode)
    assert isinstance(denominator, EventSetNode)

    return ModuloOperator(
        input_1=numerator,
        input_2=denominator,
    ).outputs["output"]


@compile
def power(
    base: EventSetOrNode,
    exponent: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(base, EventSetNode)
    assert isinstance(exponent, EventSetNode)

    return PowerOperator(
        input_1=base,
        input_2=exponent,
    ).outputs["output"]


operator_lib.register_operator(AddOperator)
operator_lib.register_operator(SubtractOperator)
operator_lib.register_operator(DivideOperator)
operator_lib.register_operator(MultiplyOperator)
operator_lib.register_operator(FloorDivOperator)
operator_lib.register_operator(ModuloOperator)
operator_lib.register_operator(PowerOperator)
