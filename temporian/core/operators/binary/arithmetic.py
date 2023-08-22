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
    """Adds two [`EventSets`][temporian.EventSet].

    Each feature in `input_1` is added to the feature in `input_2` in the same
    position.

    `input_1` and `input_2` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10, -10, 5]}
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f3": [-1, 1, 2], "f4": [1, -1, 5]},
        ...     same_sampling_as=a
        ... )

        >>> # Equivalent
        >>> c = tp.add(a, b)
        >>> c = a + b
        >>> c
        indexes: []
        features: [('add_f1_f3', int64), ('add_f2_f4', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'add_f1_f3': [ -1 101 202]
                'add_f2_f4': [ 11 -11 10]
        ...

        ```

    Cast dtypes example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10., -10., 5.]}
        ... )

        >>> # Cannot add: f1 is int64 but f2 is float64
        >>> c = a["f1"] + a["f2"]
        Traceback (most recent call last):
            ...
        ValueError: ... corresponding features should have the same dtype. ...

        >>> # Cast f1 to float
        >>> c = a["f1"].cast(tp.float64) + a["f2"]
        >>> c
        indexes: []
        features: [('add_f1_f2', float64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'add_f1_f2': [ 10. 90. 205.]
        ...

        ```

    Resample example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"fa": [1, 2, 3]},
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[-1, 1.5, 3, 5],
        ...     features={"fb": [-10, 15, 30, 50]},
        ... )

        >>> # Cannot add different samplings
        >>> c = a + b
        Traceback (most recent call last):
            ...
        ValueError: ... should have the same sampling. ...

        >>> # Resample a to match b timestamps
        >>> c = a.resample(b) + b
        >>> c
        indexes: []
        features: [('add_fa_fb', int64)]
        events:
            (4 events):
                timestamps: [-1. 1.5 3. 5. ]
                'add_fa_fb': [-10 16 33 53]
        ...

        ```

    Reindex example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3, 4],
        ...     features={
        ...         "cat": [1, 1, 2, 2],
        ...         "M": [10, 20, 30, 40]
        ...     },
        ...     indexes=["cat"]
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[1, 2, 3, 4],
        ...     features={
        ...         "cat": [1, 1, 2, 2],
        ...         "N": [10, 20, 30, 40]
        ...     },
        ... )

        >>> # Cannot add with different index (only 'a' is indexed by 'cat')
        >>> c = a + b
        Traceback (most recent call last):
            ...
        ValueError: Arguments don't have the same index. ...

        >>> # Add index 'cat' to b
        >>> b = b.add_index("cat")
        >>> # Make explicit same samplings and add
        >>> c = a + b.resample(a)
        >>> c
        indexes: [('cat', int64)]
        features: [('add_M_N', int64)]
        events:
            cat=1 (2 events):
                timestamps: [1. 2.]
                'add_M_N': [20 40]
            cat=2 (2 events):
                timestamps: [3. 4.]
                'add_M_N': [60 80]
        ...

        ```

    Args:
        input_1: First EventSet.
        input_2: Second EventSet.

    Returns:
        Sum of `input_1`'s and `input_2`'s features.
    """
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
    """Subtracts two [`EventSets`][temporian.EventSet].

    Each feature in `input_2` is subtracted from the feature in `input_1` in the
    same position.

    `input_1` and `input_2` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200]}
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f2": [10, 20, -5]},
        ...     same_sampling_as=a
        ... )

        >>> # Equivalent
        >>> c = tp.subtract(a, b)
        >>> c = a - b
        >>> c
        indexes: []
        features: [('sub_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'sub_f1_f2': [-10 80 205]
        ...

        ```

    See [`tp.add()`][temporian.add] examples to see how to match samplings,
    dtypes and index, in order to apply arithmetic operators in different
    EventSets.

    Args:
        input_1: First EventSet.
        input_2: Second EventSet.

    Returns:
        Subtraction of `input_2`'s features from `input_1`'s.
    """
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
    """Multiplies two [`EventSets`][temporian.EventSet].

    Each feature in `input_1` is multiplied by the feature in `input_2` in the
    same position.

    `input_1` and `input_2` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200]}
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f2": [10, 3, 2]},
        ...     same_sampling_as=a
        ... )

        >>> # Equivalent
        >>> c = tp.multiply(a, b)
        >>> c = a * b
        >>> c
        indexes: []
        features: [('mult_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'mult_f1_f2': [ 0 300 400]
        ...

        ```

    See [`tp.add()`][temporian.add] examples to see how to match samplings,
    dtypes and index, in order to apply arithmetic operators in different
    EventSets.

    Args:
        input_1: First EventSet.
        input_2: Second EventSet.

    Returns:
        Multiplication of `input_1`'s and `input_2`'s features.
    """
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
    """Divides two [`EventSets`][temporian.EventSet].

    Each feature in `numerator` is divided by the feature in `denominator` in
    the same position.

    This operator cannot be used in features with dtypes `int32` or `int64`.
    Cast to float before (see example) or use the
    [`tp.floordiv()`][temporian.floordiv] operator instead.

    `numerator` and `denominator` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0.0, 100.0, 200.0]}
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f2": [10.0, 20.0, 50.0]},
        ...     same_sampling_as=a
        ... )

        >>> # Equivalent
        >>> c = tp.divide(a, b)
        >>> c = a / b
        >>> c
        indexes: []
        features: [('div_f1_f2', float64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'div_f1_f2': [0. 5. 4.]
        ...

        ```

    Casting integer features:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200]}
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f2": [10, 20, 50]},
        ...     same_sampling_as=a
        ... )

        >>> # Cannot divide int64 features
        >>> c = a / b
        Traceback (most recent call last):
            ...
        ValueError: Cannot use the divide operator on feature f1 of type int64. ...

        >>> # Cast to tp.float64 or tp.float32 before
        >>> c = a.cast(tp.float64) / b.cast(tp.float64)
        >>> c
        indexes: []
        features: [('div_f1_f2', float64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'div_f1_f2': [0. 5. 4.]
        ...

        ```

    See [`tp.add()`][temporian.add] examples to see how to match samplings,
    dtypes and index, in order to apply arithmetic operators in different
    EventSets.


    Args:
        numerator: Numerator EventSet.
        denominator: Denominator EventSet.

    Returns:
        Division of `numerator`'s features by `denominator`'s features.
    """
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
    """Divides two [`EventSets`][temporian.EventSet] and takes the floor of the
    result.

    I.e. computes `numerator // denominator`.

    Each feature in `numerator` is divided by the feature in `denominator` in
    the same position.

    `numerator` and `denominator` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200]}
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f2": [10, 3, 150]},
        ...     same_sampling_as=a
        ... )

        >>> # Equivalent
        >>> c = tp.floordiv(a, b)
        >>> c = a // b
        >>> c
        indexes: []
        features: [('floordiv_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'floordiv_f1_f2': [ 0 33 1]
        ...

        ```

    See [`tp.add()`][temporian.add] examples to see how to match samplings,
    dtypes and index, in order to apply arithmetic operators in different
    EventSets.

    Args:
        numerator: Numerator EventSet.
        denominator: Denominator EventSet.

    Returns:
        Integer division of `numerator`'s features by `denominator`'s features.
    """
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
    """Computes modulo or remainder of division between two
    [`EventSets`][temporian.EventSet].

    `numerator` and `denominator` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 7, 200]}
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f2": [10, 5, 150]},
        ...     same_sampling_as=a
        ... )

        >>> # Equivalent
        >>> c = tp.modulo(a, b)
        >>> c = a % b
        >>> c
        indexes: []
        features: [('mod_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'mod_f1_f2': [ 0 2 50]
        ...

        ```

    See [`tp.add()`][temporian.add] examples to see how to match samplings,
    dtypes and index, in order to apply arithmetic operators in different
    EventSets.

    Args:
        numerator: First EventSet.
        denominator: Second EventSet.

    Returns:
        New EventSet with the remainder of the integer division.
    """
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
    """Computes elements of the base raised to the elements of the exponent
    [`EventSets`][temporian.EventSet].

    `base` and `exponent` must have the same sampling and the same number of
    features.

    `base` and `exponent` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [5, 2, 4]}
        ... )
        >>> b = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f2": [0, 3, 2]},
        ...     same_sampling_as=a
        ... )

        >>> # Equivalent
        >>> c = tp.power(a, b)
        >>> c = a ** b
        >>> c
        indexes: []
        features: [('pow_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'pow_f1_f2': [ 1 8 16]
        ...

        ```

    See [`tp.add()`][temporian.add] examples to see how to match samplings,
    dtypes and index, in order to apply arithmetic operators in different
    EventSets.

    Args:
        base: First EventSet.
        exponent: Second EventSet.

    Returns:
        New EventSet with the result of the power operation.
    """
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
