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

SCALAR = Union[float, int]


class AddScalarOperator(BaseScalarOperator):
    DEF_KEY = "ADDITION_SCALAR"


class SubtractScalarOperator(BaseScalarOperator):
    DEF_KEY = "SUBTRACTION_SCALAR"


class MultiplyScalarOperator(BaseScalarOperator):
    DEF_KEY = "MULTIPLICATION_SCALAR"


class FloorDivScalarOperator(BaseScalarOperator):
    DEF_KEY = "FLOORDIV_SCALAR"


class ModuloScalarOperator(BaseScalarOperator):
    DEF_KEY = "MODULO_SCALAR"


class PowerScalarOperator(BaseScalarOperator):
    DEF_KEY = "POWER_SCALAR"


class DivideScalarOperator(BaseScalarOperator):
    DEF_KEY = "DIVISION_SCALAR"

    def __init__(
        self,
        input: EventSetNode,
        value: Union[float, int],
        is_value_first: bool = False,
    ):
        super().__init__(input, value, is_value_first)

        for feat in input.schema.features:
            if feat.dtype in [DType.INT32, DType.INT64]:
                raise ValueError(
                    "Cannot use the divide operator on feature "
                    f"{feat.name} of type {feat.dtype}. Cast to a "
                    "floating point type or use floordiv operator (//)."
                )


@compile
def add_scalar(
    input: EventSetOrNode,
    value: Union[float, int],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return AddScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


@compile
def subtract_scalar(
    minuend: Union[EventSetOrNode, SCALAR],
    subtrahend: Union[EventSetOrNode, SCALAR],
) -> EventSetOrNode:
    scalars_types = (float, int)

    if isinstance(minuend, EventSetNode) and isinstance(
        subtrahend, scalars_types
    ):
        assert isinstance(minuend, EventSetNode)

        return SubtractScalarOperator(
            input=minuend,
            value=subtrahend,
            is_value_first=False,
        ).outputs["output"]

    if isinstance(minuend, scalars_types) and isinstance(
        subtrahend, EventSetNode
    ):
        assert isinstance(subtrahend, EventSetNode)

        return SubtractScalarOperator(
            input=subtrahend,
            value=minuend,
            is_value_first=True,
        ).outputs["output"]

    raise ValueError(
        "Invalid input types for subtract_scalar. "
        "Expected (EventSetOrNode, SCALAR) or (SCALAR, EventSetOrNode), "
        f"got ({type(minuend)}, {type(subtrahend)})."
    )


@compile
def multiply_scalar(
    input: EventSetOrNode,
    value: Union[float, int],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return MultiplyScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


@compile
def divide_scalar(
    numerator: Union[EventSetOrNode, SCALAR],
    denominator: Union[EventSetOrNode, SCALAR],
) -> EventSetOrNode:
    scalars_types = (float, int)

    if isinstance(numerator, EventSetNode) and isinstance(
        denominator, scalars_types
    ):
        assert isinstance(numerator, EventSetNode)

        return DivideScalarOperator(
            input=numerator,
            value=denominator,
            is_value_first=False,
        ).outputs["output"]

    if isinstance(numerator, scalars_types) and isinstance(
        denominator, EventSetNode
    ):
        assert isinstance(denominator, EventSetNode)

        return DivideScalarOperator(
            input=denominator,
            value=numerator,
            is_value_first=True,
        ).outputs["output"]

    raise ValueError(
        "Invalid input types for divide_scalar. "
        "Expected (EventSetOrNode, SCALAR) or (SCALAR, EventSetOrNode), "
        f"got ({type(numerator)}, {type(denominator)})."
    )


@compile
def floordiv_scalar(
    numerator: Union[EventSetOrNode, SCALAR],
    denominator: Union[EventSetOrNode, SCALAR],
) -> EventSetOrNode:
    scalars_types = (float, int)

    if isinstance(numerator, EventSetNode) and isinstance(
        denominator, scalars_types
    ):
        assert isinstance(numerator, EventSetNode)

        return FloorDivScalarOperator(
            input=numerator,
            value=denominator,
            is_value_first=False,
        ).outputs["output"]

    if isinstance(numerator, scalars_types) and isinstance(
        denominator, EventSetNode
    ):
        assert isinstance(denominator, EventSetNode)

        return FloorDivScalarOperator(
            input=denominator,
            value=numerator,
            is_value_first=True,
        ).outputs["output"]

    raise ValueError(
        "Invalid input types for floordiv_scalar. "
        "Expected (EventSetOrNode, SCALAR) or (SCALAR, EventSetOrNode), "
        f"got ({type(numerator)}, {type(denominator)})."
    )


@compile
def modulo_scalar(
    numerator: Union[EventSetOrNode, SCALAR],
    denominator: Union[EventSetOrNode, SCALAR],
) -> EventSetOrNode:
    scalar_types = (float, int)

    if isinstance(numerator, EventSetNode) and isinstance(
        denominator, scalar_types
    ):
        assert isinstance(numerator, EventSetNode)

        return ModuloScalarOperator(
            input=numerator,
            value=denominator,
            is_value_first=False,
        ).outputs["output"]

    if isinstance(numerator, scalar_types) and isinstance(
        denominator, EventSetNode
    ):
        assert isinstance(denominator, EventSetNode)

        return ModuloScalarOperator(
            input=denominator,
            value=numerator,
            is_value_first=True,
        ).outputs["output"]

    raise ValueError(
        "Invalid input types for modulo_scalar. "
        "Expected (EventSetOrNode, SCALAR) or (SCALAR, EventSetOrNode), "
        f"got ({type(numerator)}, {type(denominator)})."
    )


@compile
def power_scalar(
    base: Union[EventSetOrNode, SCALAR],
    exponent: Union[EventSetOrNode, SCALAR],
) -> EventSetOrNode:
    scalar_types = (float, int)

    if isinstance(base, EventSetNode) and isinstance(exponent, scalar_types):
        assert isinstance(base, EventSetNode)

        return PowerScalarOperator(
            input=base,
            value=exponent,
            is_value_first=False,
        ).outputs["output"]

    if isinstance(base, scalar_types) and isinstance(exponent, EventSetNode):
        assert isinstance(exponent, EventSetNode)

        return PowerScalarOperator(
            input=exponent,
            value=base,
            is_value_first=True,
        ).outputs["output"]

    raise ValueError(
        "Invalid input types for power_scalar. "
        "Expected (EventSetOrNode, SCALAR) or (SCALAR, EventSetOrNode), "
        f"got ({type(base)}, {type(exponent)})."
    )


operator_lib.register_operator(SubtractScalarOperator)
operator_lib.register_operator(AddScalarOperator)
operator_lib.register_operator(MultiplyScalarOperator)
operator_lib.register_operator(DivideScalarOperator)
operator_lib.register_operator(FloorDivScalarOperator)
operator_lib.register_operator(ModuloScalarOperator)
operator_lib.register_operator(PowerScalarOperator)
