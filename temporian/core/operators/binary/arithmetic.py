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
from temporian.core.data.node import Node
from temporian.core.data.dtype import DType
from temporian.core.operators.binary.base import BaseBinaryOperator


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
        input_1: Node,
        input_2: Node,
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
    input_1: Node,
    input_2: Node,
) -> Node:
    """Adds two nodes.

    Each feature in `input_1` is added to the feature in `input_2` in the same
    position.

    `input_1` and `input_2` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10, -10, 5]}
        ... )
        >>> source = evset.node()
        >>> a = source[["f1", "f2"]]
        >>> b = source[["f2", "f1"]]  # Reverse order

        >>> # Equivalent
        >>> c = tp.add(a, b)
        >>> c = a + b

        >>> c.run({source: evset})
        indexes: []
        features: [('add_f1_f2', int64), ('add_f2_f1', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'add_f1_f2': [ 10 90 205]
                'add_f2_f1': [ 10 90 205]
        ...

        ```

    Cast dtypes example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [-30.0, 10.0, 5.0]}
        ... )
        >>> source = evset.node()

        >>> # a is int64 but b is float64
        >>> a = source["f1"]
        >>> b = source["f2"]

        >>> # Cannot add different dtypes
        >>> c = a + b
        Traceback (most recent call last):
            ...
        ValueError: ... corresponding features should have the same dtype. ...

        >>> # Cast f1 to float
        >>> c = tp.cast(a, tp.float64) + b
        >>> c.run({source: evset})
        indexes: []
        features: [('add_f1_f2', float64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'add_f1_f2': [-30. 110. 205.]
        ...

        ```

    Resample example:
        ```python
        >>> a_evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"fa": [1, 2, 3]},
        ...     name="sample_a"
        ... )
        >>> b_evset = tp.event_set(
        ...     timestamps=[-1, 1.5, 3, 5],
        ...     features={"fb": [-10, 15, 30, 50]},
        ...     name="sample_b"
        ... )
        >>> a = a_evset.node()
        >>> b = b_evset.node()

        >>> # Cannot add different samplings
        >>> c = a + b
        Traceback (most recent call last):
            ...
        ValueError: ... should have the same sampling. ...

        >>> # Resample a to match b timestamps
        >>> a_resampled = tp.resample(a, sampling=b)
        >>> c = a_resampled + b
        >>> c.run([a_evset, b_evset])
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
        >>> a_evset = tp.event_set(
        ...     timestamps=[1, 2, 3, 4],
        ...     features={
        ...         "f1": [1, 1, 1, 1],
        ...         "f2": [1, 2, 3, 4],
        ...         "f3": [10, 20, 30, 40]
        ...     },
        ...     indexes=["f1", "f2"]
        ... )
        >>> b_evset = tp.event_set(
        ...     timestamps=[1, 2, 3, 4],
        ...     features={
        ...         "f1": [1, 1, 1, 1],
        ...         "f2": [5, 6, 7, 8],
        ...         "f3": [10, 20, 30, 40]
        ...     },
        ...     indexes=["f1", "f2"]
        ... )
        >>> a = a_evset.node()
        >>> b = b_evset.node()

        >>> # Get f3 per f1
        >>> a_f1 = tp.drop_index(a, "f2")
        >>> b_f1 = tp.drop_index(b, "f2")

        >>> # We also need to explicit same samplings
        >>> b_f1 = tp.resample(b_f1, a_f1)

        >>> c = a_f1["f3"] + b_f1["f3"]

        >>> c.run({a: a_evset, b: b_evset})
        indexes: [('f1', int64)]
        ...
                timestamps: [1. 2. 3. 4.]
                'add_f3_f3': [20 40 60 80]
        ...

        ```

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Sum of `input_1`'s and `input_2`'s features.
    """
    return AddOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def subtract(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Subtracts two nodes.

    Each feature in `input_2` is subtracted from the feature in `input_1` in the
    same position.

    `input_1` and `input_2` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10, 20, -5]}
        ... )
        >>> source = evset.node()
        >>> a = source["f1"]
        >>> b = source["f2"]

        >>> # Equivalent
        >>> c = tp.subtract(a, b)
        >>> c = a - b

        >>> c.run({source: evset})
        indexes: []
        features: [('sub_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'sub_f1_f2': [-10 80 205]
        ...

        ```

    See [`tp.add()`](../add) examples to see how to match samplings, dtypes and
    index, in order to apply arithmetic operators in different nodes.

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Subtraction of `input_2`'s features from `input_1`'s.
    """
    return SubtractOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def multiply(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Multiplies two nodes.

    Each feature in `input_1` is multiplied by the feature in `input_2` in the
    same position.

    `input_1` and `input_2` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10, 3, 2]}
        ... )
        >>> source = evset.node()
        >>> a = source["f1"]
        >>> b = source["f2"]

        >>> # Equivalent
        >>> c = tp.multiply(a, b)
        >>> c = a * b

        >>> c.run({source: evset})
        indexes: []
        features: [('mult_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'mult_f1_f2': [ 0 300 400]
        ...

        ```

    See [`tp.add()`](../add) examples to see how to match samplings, dtypes and
    index, in order to apply arithmetic operators in different nodes.

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Multiplication of `input_1`'s and `input_2`'s features.
    """
    return MultiplyOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def divide(
    numerator: Node,
    denominator: Node,
) -> Node:
    """Divides two nodes.

    Each feature in `numerator` is divided by the feature in `denominator` in
    the same position.

    This operator cannot be used in features with dtypes `int32` or `int64`.
    Cast to float before (see example) or use the [`tp.floordiv`](../floordiv)
    operator instead.

    `numerator` and `denominator` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0.0, 100.0, 200.0], "f2": [10.0, 20.0, 50.0]}
        ... )
        >>> source = evset.node()
        >>> a = source["f1"]
        >>> b = source["f2"]

        >>> # Equivalent
        >>> c = tp.divide(a, b)
        >>> c = a / b

        >>> c.run({source: evset})
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
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10, 20, 50]}
        ... )
        >>> source = evset.node()

        >>> # Both features are int64
        >>> a = source["f1"]
        >>> b = source["f2"]

        >>> # Cannot divide int64 features
        >>> c = a / b
        Traceback (most recent call last):
            ...
        ValueError: Cannot use the divide operator on feature f1 of type int64. ...

        >>> # Cast to tp.float64 or tp.float32 before
        >>> c = tp.cast(a, float) / tp.cast(b, float)
        >>> c.run({source: evset})
        indexes: []
        features: [('div_f1_f2', float64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'div_f1_f2': [0. 5. 4.]
        ...

        ```

    See [`tp.add()`](../add) examples to see how to match samplings, dtypes and
    index, in order to apply arithmetic operators in different nodes.


    Args:
        numerator: Numerator node.
        denominator: Denominator node.

    Returns:
        Division of `numerator`'s features by `denominator`'s features.
    """
    return DivideOperator(
        input_1=numerator,
        input_2=denominator,
    ).outputs["output"]


@compile
def floordiv(
    numerator: Node,
    denominator: Node,
) -> Node:
    """Divides two nodes and takes the floor of the result.

    I.e. computes numerator//denominator.

    Each feature in `numerator` is divided by the feature in `denominator` in
    the same position.

    `numerator` and `denominator` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [10, 3, 150]}
        ... )
        >>> source = evset.node()
        >>> a = source["f1"]
        >>> b = source["f2"]

        >>> # Equivalent
        >>> c = tp.floordiv(a, b)
        >>> c = a // b

        >>> c.run({source: evset})
        indexes: []
        features: [('floordiv_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'floordiv_f1_f2': [ 0 33 1]
        ...

        ```

    See [`tp.add()`](../add) examples to see how to match samplings, dtypes and
    index, in order to apply arithmetic operators in different nodes.

    Args:
        numerator: Numerator node.
        denominator: Denominator node.

    Returns:
        Integer division of `numerator`'s features by `denominator`'s features.
    """
    return FloorDivOperator(
        input_1=numerator,
        input_2=denominator,
    ).outputs["output"]


@compile
def modulo(
    numerator: Node,
    denominator: Node,
) -> Node:
    """Computes modulo or remainder of division between two
    nodes.

    `numerator` and `denominator` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 7, 200], "f2": [10, 5, 150]}
        ... )
        >>> source = evset.node()
        >>> a = source["f1"]
        >>> b = source["f2"]

        >>> # Equivalent
        >>> c = tp.modulo(a, b)
        >>> c = a % b

        >>> c.run({source: evset})
        indexes: []
        features: [('mod_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'mod_f1_f2': [ 0 2 50]
        ...

        ```

    See [`tp.add()`](../add) examples to see how to match samplings, dtypes and
    index, in order to apply arithmetic operators in different nodes.

    Args:
        numerator: First node.
        denominator: Second node.

    Returns:
        New node with the remainder of the integer division
    """
    return ModuloOperator(
        input_1=numerator,
        input_2=denominator,
    ).outputs["output"]


@compile
def power(
    base: Node,
    exponent: Node,
) -> Node:
    """Computes elements of the base raised to the elements of the exponent.

    `base` and `exponent` must have the same sampling and the same number of
    features.

    `base` and `exponent` must have the same sampling, index,
    number of features and dtype for the features in the same positions.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [5, 2, 4], "f2": [0, 3, 2]}
        ... )
        >>> source = evset.node()
        >>> a = source["f1"]
        >>> b = source["f2"]

        >>> # Equivalent
        >>> c = tp.power(a, b)
        >>> c = a ** b

        >>> c.run({source: evset})
        indexes: []
        features: [('pow_f1_f2', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'pow_f1_f2': [ 1 8 16]
        ...

        ```

    See [`tp.add()`](../add) examples to see how to match samplings, dtypes and
    index, in order to apply arithmetic operators in different nodes.

    Args:
        base: First node.
        exponent: Second node.

    Returns:
        New node with the remainder of the integer division
    """
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
