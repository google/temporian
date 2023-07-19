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
    """Adds a scalar value to an [`EventSet`][temporian.EventSet].

    `value` is added to each item in each feature in `input`.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10, -10, 5]}
        ... )

        >>> # Equivalent
        >>> b = tp.add_scalar(a, 3)
        >>> b = a + 3
        >>> b
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [ 3 103 203]
                'f2': [13 -7 8]
        ...

        ```

    Args:
        input: EventSetNode to add a scalar to.
        value: Scalar value to add to the input.

    Returns:
        Addition of `input` and `value`.
    """
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
    """Subtracts an [`EventSet`][temporian.EventSet] and a scalar value.

    Each item in each feature in the EventSet is subtracted with the scalar
    value.

    Either `minuend` or `subtrahend` should be a scalar value, but not both. If
    looking to subtract two EventSets, use the
    [`tp.subtract()`][temporian.subtract] operator instead.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10, -10, 5]}
        ... )

        >>> # Equivalent
        >>> b = tp.subtract_scalar(a, 3)
        >>> b = a - 3
        >>> b
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [ -3  97 197]
                'f2': [ 7 -13   2]
        ...

        >>> # Equivalent
        >>> c = tp.subtract_scalar(3, a)
        >>> c = 3 - a
        >>> c
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [ 3  -97 -197]
                'f2': [-7 13  -2]
        ...

        ```

    Args:
        minuend: EventSet or scalar value being subtracted from.
        subtrahend: EventSet or scalar number being subtracted.

    Returns:
        EventSet with the difference between the minuend and subtrahend.
    """
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
    """Multiplies an [`EventSet`][temporian.EventSet] by a scalar value.

    Each item in each feature in `input` is multiplied by `value`.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10, -10, 5]}
        ... )

        >>> # Equivalent
        >>> b = tp.multiply_scalar(a, 2)
        >>> b = a * 2
        >>> b
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [ 0 200 400]
                'f2': [ 20 -20 10]
        ...

        ```

    Args:
        input: EventSet to multiply.
        value: Scalar value to multiply the input by.

    Returns:
        Integer division of `input` and `value`.
    """
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
    """Divides an [`EventSet`][temporian.EventSet] and a scalar value.

    Each item in each feature in the EventSet is divided with the scalar value.

    Either `numerator` or `denominator` should be a scalar value, but not both.
    If looking to divide two EventSets, use the
    [`tp.divide()`][temporian.divide] operator instead.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0., 100., 200.], "f2": [10., -10., 5.]}
        ... )

        >>> # Equivalent
        >>> b = tp.divide_scalar(a, 2)
        >>> b = a / 2
        >>> b
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [ 0. 50. 100.]
                'f2': [ 5. -5. 2.5]
        ...

        >>> # Equivalent
        >>> c = tp.divide_scalar(1000, a)
        >>> c = 1000 / a
        >>> c
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [inf 10. 5.]
                'f2': [ 100. -100. 200.]
        ...

        ```

    Args:
        numerator: Numerator EventSet or value.
        denominator: Denominator EventSet or value.

    Returns:
        Division of `numerator` and `denominator`.
    """
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
    """Divides an [`EventSet`][temporian.EventSet] and a scalar and takes the
    result's floor.

    Each item in each feature in the EventSet is divided with the scalar value.

    Either `numerator` or `denominator` should be a scalar value, but not both.
    If looking to floordiv two EventSet, use the
    [`tp.floordiv()`][temporian.floordiv] operator instead.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [1, 100, 200], "f2": [10., -10., 5.]}
        ... )

        >>> # Equivalent
        >>> b = tp.floordiv_scalar(a, 3)
        >>> b = a // 3
        >>> b
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [ 0 33 66]
                'f2': [ 3. -4. 1.]
        ...

        >>> # Equivalent
        >>> c = tp.floordiv_scalar(300, a)
        >>> c = 300 // a
        >>> c
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [300 3 1]
                'f2': [ 30. -30. 60.]
        ...

        ```

    Args:
        numerator: Numerator EventSet or value.
        denominator: Denominator EventSet or value.

    Returns:
        Integer division of `numerator` and `denominator`.
    """
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
    """Remainder of the division of numerator by denominator
    [`EventSets`][temporian.EventSet].

    Either `numerator` or `denominator` should be a scalar value, but not both.
    For the operation between two EventSets, use the
    [`tp.modulo()`][temporian.modulo] operator instead.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [1, 100, 200], "f2": [10., -10., 5.]}
        ... )

        >>> # Equivalent
        >>> b = tp.modulo_scalar(a, 3)
        >>> b = a % 3
        >>> b
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [1 1 2]
                'f2': [1. 2. 2.]
        ...

        >>> # Equivalent
        >>> c = tp.floordiv_scalar(300, a)
        >>> c = 300 % a
        >>> c
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [ 0 0 100]
                'f2': [ 0. -0. 0.]
        ...

        ```

    Args:
        numerator: EventSet or scalar to divide.
        denominator: EventSet or scalar to divide by.

    Returns:
        Remainder of the integer division.
    """
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
    """Raise the base to the exponent (`base ** exponent`)
    [`EventSets`][temporian.EventSet].

    Either `base` or `exponent` should be a scalar value, but not both.
    For the operation between two EventSets, use the
    [`tp.power()`][temporian.power] operator instead.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 2, 3], "f2": [1., 2., 3.]}
        ... )

        >>> # Equivalent
        >>> b = tp.power_scalar(a, 3)
        >>> b = a ** 3
        >>> b
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [ 0 8 27]
                'f2': [ 1. 8. 27.]
        ...

        >>> # Equivalent
        >>> c = tp.power_scalar(3, a)
        >>> c = 3 ** a
        >>> c
        indexes: ...
                timestamps: [1. 2. 3.]
                'f1': [ 1 9 27]
                'f2': [ 3. 9. 27.]
        ...

        ```

    Args:
        base: EventSet or scalar to raise to the exponent.
        exponent: EventSet or scalar for the exponent.

    Returns:
        Base values raised to the exponent.
    """
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
