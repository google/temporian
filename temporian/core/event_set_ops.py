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

# pylint: disable=import-outside-toplevel

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from temporian.core.data.duration import Duration

if TYPE_CHECKING:
    from temporian.core.operators.map import MapFunction
    from temporian.core.typing import (
        EventSetOrNode,
        IndexKeyList,
        TargetDtypes,
        WindowLength,
    )

T_SCALAR = (int, float)


class EventSetOperations:
    """Mixin class for EventSet-like classes.

    Defines the methods that can be called on both EventSets and EventSetNodes
    interchangeably.
    """

    @property
    def _clsname(self) -> str:
        """Shortcut that returns the class' name."""
        return self.__class__.__name__

    #################
    # MAGIC METHODS #
    #################

    def __getitem__(self, feature_names: Union[str, List[str]]):
        """Creates an EventSet with a subset of the features."""

        from temporian.core.operators.select import select

        return select(self, feature_names)

    def __setitem__(self, feature_names: Any, value: Any) -> None:
        """Fails, features cannot be assigned."""

        raise TypeError(
            f"Cannot assign features to an existing {self._clsname}. New"
            f" {self._clsname}s should be created instead. Check out the"
            " `tp.glue()` operator to combine features from several"
            f" {self._clsname}s."
        )

    def __bool__(self) -> None:
        """Catches bool evaluation with an error message."""

        # Called on "if node" or "if evset" conditions
        # TODO: modify to similar numpy msg if we implement .any() or .all()
        raise ValueError(
            f"The truth value of a {self._clsname} is ambiguous. Check"
            f" condition element-wise or use the `{self._clsname}.cast()`"
            " operator to convert to boolean."
        )

    def _raise_error(
        self, op_name: str, other: Any, allowed_types: str
    ) -> None:
        """Raises an error message.

        This utility method is used in operator implementations, e.g., +, - *.
        """

        raise ValueError(
            f"Cannot use operator '{op_name}' on {self._clsname} and"
            f" {type(other)} objects. Only {self._clsname} or values of type"
            f" ({allowed_types}) are supported."
        )

    def __ne__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes not equal (`self != other`) element-wise with another
        [`EventSet`][temporian.EventSet] or a scalar value.

        If an EventSet, each feature in `self` is compared element-wise to
        the feature in `other` in the same position. `self` and `other`
        must have the same sampling and the same number of features.

        If a scalar value, each item in each feature in `self` is compared to
        `other`.

        Note that it will always return True on NaNs (even if both are).

        Example with EventSet:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200]}
            ... )
            >>> b = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f2": [-10, 100, 5]},
            ...     same_sampling_as=a
            ... )

            >>> c = a != b
            >>> c
            indexes: []
            features: [('ne_f1_f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'ne_f1_f2': [ True False True]
            ...

            ```

        Example with scalar value:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
            ... )

            >>> b = a != 100
            >>> b
            indexes: []
            features: [('f1', bool_), ('f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'f1': [ True False True]
                    'f2': [ True False True]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the comparison.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import not_equal

            return not_equal(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR + (bool, str)):
            from temporian.core.operators.scalar import not_equal_scalar

            return not_equal_scalar(input=self, value=other)

        self._raise_error("ne", other, "int,float,bool,str")
        assert False

    def __add__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Adds an [`EventSet`][temporian.EventSet] or a scalar value to
        `self` element-wise.

        If an EventSet, each feature in `self` is added to the feature in
        `other` in the same position. `self` and `other` must have the same
        sampling, index, number of features and dtype for the features in the
        same positions.

        If a scalar, `other` is added to each item in each feature in `self`.

        Example with EventSet:
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

        Example with scalar value:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200], "f2": [10, -10, 5]}
            ... )

            >>> b = a + 3
            >>> b
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 3 103 203]
                    'f2': [13 -7 8]
            ...

            >>> b = 3 + a
            >>> b
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 3 103 203]
                    'f2': [13 -7 8]
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
            other: EventSet or scalar value.

        Returns:
            Result of the operation.
        """
        # TODO: In this and other operants, factor code and add support for
        # swapping operators (e.g. a+1, a+b, 1+a).

        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import add

            return add(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import add_scalar

            return add_scalar(input=self, value=other)

        self._raise_error("add", other, "int,float")
        assert False

    def __radd__(self, other: Any):
        return self.__add__(other)

    def __sub__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Subtracts an [`EventSet`][temporian.EventSet] or a scalar value from
        `self` element-wise.

        If an EventSet, each feature in `self` is subtracted from the feature in
        `other` in the same position. `self` and `other` must have the same
        sampling, index, number of features and dtype for the features in the
        same positions.

        If a scalar, `other` is subtracted from each item in each feature in
        `self`.

        See examples in [`EventSet.__add__()`][temporian.EventSet.__add__] to
        see how to match samplings, dtypes and index, in order to apply
        arithmetic operators in different EventSets.

        Example with EventSet:
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

        Example with scalar value:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200], "f2": [10, -10, 5]}
            ... )

            >>> b = a - 3
            >>> b
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ -3  97 197]
                    'f2': [ 7 -13   2]
            ...

            >>> c = 3 - a
            >>> c
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 3  -97 -197]
                    'f2': [-7 13  -2]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the operation.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import subtract

            return subtract(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import subtract_scalar

            return subtract_scalar(minuend=self, subtrahend=other)

        self._raise_error("subtract", other, "int,float")
        assert False

    def __rsub__(self, other: Any):
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import subtract_scalar

            return subtract_scalar(minuend=other, subtrahend=self)

        self._raise_error("subtract", other, "int,float")
        assert False

    def __mul__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Multiplies an [`EventSet`][temporian.EventSet] or a scalar value with
        `self` element-wise.

        If an EventSet, each feature in `self` is multiplied with the feature in
        `other` in the same position. `self` and `other` must have the same
        sampling, index, number of features and dtype for the features in the
        same positions.

        If a scalar, each item in each feature in `self` is multiplied with
        `other`.

        See examples in [`EventSet.__add__()`][temporian.EventSet.__add__] to
        see how to match samplings, dtypes and index, in order to apply
        arithmetic operators in different EventSets.

        Example with EventSet:
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

        Example with scalar value:
             ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200], "f2": [10, -10, 5]}
            ... )

            >>> b = a * 2
            >>> b
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 0 200 400]
                    'f2': [ 20 -20 10]
            ...

            >>> b = 2 * a
            >>> b
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 0 200 400]
                    'f2': [ 20 -20 10]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the operation.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import multiply

            return multiply(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import multiply_scalar

            return multiply_scalar(input=self, value=other)

        self._raise_error("multiply", other, "int,float")
        assert False

    def __rmul__(self, other: Any):
        return self.__mul__(other)

    def __neg__(self: EventSetOrNode) -> EventSetOrNode:
        """Negates an [`EventSet`][temporian.EventSet] element-wise.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2],
            ...     features={"M": [1, -5], "N": [-1.0, 5.5]},
            ... )
            >>> -a
            indexes: ...
                    'M': [-1  5]
                    'N': [ 1.  -5.5]
            ...

            ```

        Returns:
            Negated EventSet.
        """
        from temporian.core.operators.scalar import multiply_scalar

        return multiply_scalar(input=self, value=-1)

    def __invert__(self: EventSetOrNode) -> EventSetOrNode:
        """Inverts a boolean [`EventSet`][temporian.EventSet] element-wise.

        Swaps False <-> True.

        Does not work on integers, they should be cast to
        [`tp.bool_`][temporian.bool_] beforehand, using
        [`EventSet.cast()`][temporian.EventSet.cast].

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2],
            ...     features={"M": [1, 5], "N": [1.0, 5.5]},
            ... )
            >>> # Boolean EventSet
            >>> b = a < 2
            >>> b
            indexes: ...
                    'M': [ True False]
                    'N': [ True False]
            ...

            >>> # Inverted EventSet
            >>> c = ~b
            >>> c
            indexes: ...
                    'M': [False True]
                    'N': [False True]
            ...

            ```

        Returns:
            Inverted EventSet.
        """
        from temporian.core.operators.unary import invert

        return invert(input=self)

    def __abs__(self):
        from temporian.core.operators.unary import abs

        return abs(input=self)

    def __truediv__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Divides `self` by an [`EventSet`][temporian.EventSet] or a scalar
        value element-wise.

        If an EventSet, each feature in `self` is divided by the feature in
        `other` in the same position. `self` and `other` must have the same
        sampling, index, number of features and dtype for the features in the
        same positions.

        If a scalar, each item in each feature in `self` is divided by `other`.

        This operator cannot be used in features with dtypes `int32` or `int64`.
        Cast to float before (see example) or use
        [`EventSet.__floordiv__()`][temporian.EventSet.__floordiv__] instead.

        See examples in [`EventSet.__add__()`][temporian.EventSet.__add__] to
        see how to match samplings, dtypes and index, in order to apply
        arithmetic operators in different EventSets.

        Example with EventSet:
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

        Example casting integer features:
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

        Example with scalar value:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0., 100., 200.], "f2": [10., -10., 5.]}
            ... )

            >>> b = a / 2
            >>> b
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 0. 50. 100.]
                    'f2': [ 5. -5. 2.5]
            ...

            >>> c = 1000 / a
            >>> c
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [inf 10. 5.]
                    'f2': [ 100. -100. 200.]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the operation.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import divide

            return divide(numerator=self, denominator=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import divide_scalar

            return divide_scalar(numerator=self, denominator=other)

        self._raise_error("divide", other, "(int,float)")
        assert False

    def __rtruediv__(self, other: Any):
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import divide_scalar

            return divide_scalar(numerator=other, denominator=self)

        self._raise_error("divide", other, "(int,float)")
        assert False

    def __floordiv__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Divides `self` by an [`EventSet`][temporian.EventSet] or a scalar
        value and takes the floor of the result, element-wise.

        If an EventSet, each feature in `self` is divided by the feature in
        `other` in the same position. `self` and `other` must have the same
        sampling, index, number of features and dtype for the features in the
        same positions.

        If a scalar, each item in each feature in `self` is divided by `other`.

        See examples in [`EventSet.__add__()`][temporian.EventSet.__add__] to
        see how to match samplings, dtypes and index, in order to apply
        arithmetic operators in different EventSets.

        Example with EventSet:
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

        Example with scalar value:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [1, 100, 200], "f2": [10., -10., 5.]}
            ... )

            >>> b = a // 3
            >>> b
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 0 33 66]
                    'f2': [ 3. -4. 1.]
            ...

            >>> c = 300 // a
            >>> c
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [300 3 1]
                    'f2': [ 30. -30. 60.]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the operation.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import floordiv

            return floordiv(numerator=self, denominator=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import floordiv_scalar

            return floordiv_scalar(numerator=self, denominator=other)

        self._raise_error("floor_divide", other, "(int,float)")
        assert False

    def __rfloordiv__(self, other: Any):
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import floordiv_scalar

            return floordiv_scalar(numerator=other, denominator=self)

        self._raise_error("floor_divide", other, "(int,float)")
        assert False

    def __pow__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes power with another
        [`EventSet`][temporian.EventSet] or a scalar value element-wise.

        If an EventSet, each feature in `self` is raised to the feature in
        `other` in the same position. `self` and `other` must have the same
        sampling, index, number of features and dtype for the features in the
        same positions.

        If a scalar, each item in each feature in `self` is raised to
        `other`.

        See examples in [`EventSet.__add__()`][temporian.EventSet.__add__] to
        see how to match samplings, dtypes and index, in order to apply
        arithmetic operators in different EventSets.

        Example with EventSet:
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

        Example with scalar value:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 2, 3], "f2": [1., 2., 3.]}
            ... )

            >>> b = a ** 3
            >>> b
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 0 8 27]
                    'f2': [ 1. 8. 27.]
            ...

            >>> c = 3 ** a
            >>> c
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 1 9 27]
                    'f2': [ 3. 9. 27.]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the operation.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import power

            return power(base=self, exponent=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import power_scalar

            return power_scalar(base=self, exponent=other)

        self._raise_error("exponentiate", other, "(int,float)")
        assert False

    def __rpow__(self, other: Any):
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import power_scalar

            return power_scalar(base=other, exponent=self)

        self._raise_error("exponentiate", other, "(int,float)")
        assert False

    def __mod__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes modulo or remainder of division with another
        [`EventSet`][temporian.EventSet] or a scalar value.

        If an EventSet, each feature in `self` is reduced modulo the feature in
        `other` in the same position. `self` and `other` must have the same
        sampling, index, number of features and dtype for the features in the
        same positions.

        If a scalar, each item in each feature in `self` is reduced modulo
        `other`.

        See examples in [`EventSet.__add__()`][temporian.EventSet.__add__] to
        see how to match samplings, dtypes and index, in order to apply
        arithmetic operators in different EventSets.

        Example with EventSet:
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

        Example with scalar value:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [1, 100, 200], "f2": [10., -10., 5.]}
            ... )

            >>> b = a % 3
            >>> b
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [1 1 2]
                    'f2': [1. 2. 2.]
            ...

            >>> c = 300 % a
            >>> c
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'f1': [ 0 0 100]
                    'f2': [ 0. -0. 0.]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the operation.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import modulo

            return modulo(numerator=self, denominator=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import modulo_scalar

            return modulo_scalar(numerator=self, denominator=other)

        self._raise_error("compute modulo (%)", other, "(int,float)")
        assert False

    def __rmod__(self, other: Any):
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import modulo_scalar

            return modulo_scalar(numerator=other, denominator=self)

        self._raise_error("compute modulo (%)", other, "(int,float)")
        assert False

    def __gt__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes greater (`self > other`) element-wise with another
        [`EventSet`][temporian.EventSet] or a scalar value.

        If an EventSet, each feature in `self` is compared element-wise to the
        feature in `other` in the same position. `self` and `other` must have
        the same sampling and the same number of features.

        If a scalar value, each item in each feature in `self` is compared to
        `other`.

        Note that it will always return False on NaN elements.

        Example with EventSet:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200]}
            ... )
            >>> b = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f2": [-10, 100, 5]},
            ...     same_sampling_as=a
            ... )

            >>> c = a > b
            >>> c
            indexes: []
            features: [('gt_f1_f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'gt_f1_f2': [ True False True]
            ...

            ```

        Example with scalar:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
            ... )

            >>> b = a != 100
            >>> b
            indexes: []
            features: [('f1', bool_), ('f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'f1': [ True False True]
                    'f2': [ True False True]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the comparison.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import greater

            return greater(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import greater_scalar

            return greater_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def __ge__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes greater equal (`self >= other`) element-wise with another
        [`EventSet`][temporian.EventSet] or a scalar value.

        If an EventSet, each feature in `self` is compared element-wise to the
        feature in `other` in the same position. `self` and `other` must have
        the same sampling and the same number of features.

        If a scalar value, each item in each feature in `input` is compared to
        `value`.

        Note that it will always return False on NaN elements.

        Example with EventSet:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200]}
            ... )
            >>> b = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f2": [-10, 100, 5]},
            ...     same_sampling_as=a
            ... )

            >>> c = a >= b
            >>> c
            indexes: []
            features: [('ge_f1_f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'ge_f1_f2': [ True True True]
            ...

            ```

        Example with scalar:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
            ... )

            >>> b = a >= 100
            >>> b
            indexes: []
            features: [('f1', bool_), ('f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'f1': [False True True]
                    'f2': [False True False]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the comparison.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import greater_equal

            return greater_equal(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import greater_equal_scalar

            return greater_equal_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def __lt__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes less (`self < other`) element-wise with another
        [`EventSet`][temporian.EventSet] or a scalar value.

        If an EventSet, each feature in `self` is compared element-wise to the
        feature in `other` in the same position. `self` and `other` must have
        the same sampling and the same number of features.

        If a scalar value, each item in each feature in `input` is compared to
        `value`.

        Note that it will always return False on NaN elements.

        Example with EventSet:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200]}
            ... )
            >>> b = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f2": [-10, 100, 5]},
            ...     same_sampling_as=a
            ... )

            >>> c = a < b
            >>> c
            indexes: []
            features: [('lt_f1_f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'lt_f1_f2': [False False False]
            ...

            ```

        Example with scalar:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
            ... )

            >>> b = a < 100
            >>> b
            indexes: []
            features: [('f1', bool_), ('f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'f1': [ True False False]
                    'f2': [ True False True]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the comparison.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import less

            return less(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import less_scalar

            return less_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def __le__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes less equal (`self <= other`) element-wise with another
        [`EventSet`][temporian.EventSet] or a scalar value.

        If an EventSet, each feature in `self` is compared element-wise to the
        feature in `other` in the same position. `self` and `other` must have
        the same sampling and the same number of features.

        If a scalar value, each item in each feature in `input` is compared to
        `value`.

        Note that it will always return False on NaN elements.

        Example with EventSet:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200]}
            ... )
            >>> b = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f2": [-10, 100, 5]},
            ...     same_sampling_as=a
            ... )

            >>> c = a <= b
            >>> c
            indexes: []
            features: [('le_f1_f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'le_f1_f2': [False True False]
            ...

            ```

        Example with scalar:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
            ... )

            >>> b = a <= 100
            >>> b
            indexes: []
            features: [('f1', bool_), ('f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'f1': [ True True False]
                    'f2': [ True True True]
            ...

            ```

        Args:
            other: EventSet or scalar value.

        Returns:
            Result of the comparison.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import less_equal

            return less_equal(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import less_equal_scalar

            return less_equal_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def _raise_bool_error(self, boolean_op: str, other: Any) -> None:
        raise ValueError(
            f"Cannot compute '{self._clsname} {boolean_op} {type(other)}'. "
            f"Only {self._clsname}s with boolean features are supported."
        )

    def __and__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes logical and (`self & other`) element-wise with another
        [`EventSet`][temporian.EventSet].

        Each feature in `self` is compared element-wise to the feature in
        `other` in the same position.

        `self` and `other` must have the same sampling, the same number of
        features, and all feature types must be `bool` (see cast example below).

        Example:
            ```python
            >>> a = tp.event_set(timestamps=[1, 2, 3], features={"f1": [100, 150, 200]})

            >>> # Sample boolean features
            >>> b = a > 100
            >>> c = a < 200

            >>> d = b & c
            >>> d
            indexes: []
            features: [('and_f1_f1', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'and_f1_f1': [False True False]
            ...

            ```

        Example casting integer to boolean:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 1, 1], "f2": [1, 1, 0]}
            ... )
            >>> b = a.cast(bool)
            >>> c = b["f1"] & b["f2"]
            >>> c
            indexes: []
            features: [('and_f1_f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'and_f1_f2': [False True False]
            ...

            ```

        Args:
            other: EventSet with only boolean features.

        Returns:
            EventSet with result of the comparison.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import logical_and

            return logical_and(input_1=self, input_2=other)

        self._raise_bool_error("&", other)
        assert False

    def __or__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes logical or (`self | other`) element-wise with another
        [`EventSet`][temporian.EventSet].

        Each feature in `self` is compared element-wise to the feature in
        `other` in the same position.

        `self` and `other` must have the same sampling, the same number of
        features, and all feature types must be `bool`.

        See cast example in [`EventSet.__and__()`][temporian.EventSet.__and__].

        Example:
            ```python
            >>> a = tp.event_set(timestamps=[1, 2, 3], features={"f1": [100, 150, 200]})

            >>> # Sample boolean features
            >>> b = a <= 100
            >>> c = a >= 200

            >>> d = b | c
            >>> d
            indexes: []
            features: [('or_f1_f1', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'or_f1_f1': [ True False True]
            ...

            ```

        Args:
            other: EventSet with only boolean features.

        Returns:
            EventSet with result of the comparison.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import logical_or

            return logical_or(input_1=self, input_2=other)

        self._raise_bool_error("|", other)
        assert False

    def __xor__(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Computes logical xor (`self ^ other`) element-wise with another
        [`EventSet`][temporian.EventSet].

        Each feature in `self` is compared element-wise to the feature in
        `other` in the same position.

        `self` and `other` must have the same sampling, the same number of
        features, and all feature types must be `bool`.

        See cast example in [`EventSet.__and__()`][temporian.EventSet.__and__].

        Example:
            ```python
            >>> a = tp.event_set(timestamps=[1, 2, 3], features={"f1": [100, 150, 200]})

            >>> # Sample boolean features
            >>> b = a > 100
            >>> c = a < 200

            >>> d = b ^ c
            >>> d
            indexes: []
            features: [('xor_f1_f1', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'xor_f1_f1': [ True False True]
            ...

            ```

        Args:
            other: EventSet with only boolean features.

        Returns:
            EventSet with result of the comparison.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import logical_xor

            return logical_xor(input_1=self, input_2=other)

        self._raise_bool_error("^", other)
        assert False

    #############
    # OPERATORS #
    #############

    def abs(
        self: EventSetOrNode,
    ) -> EventSetOrNode:
        """Gets the absolute value of an [`EventSet`][temporian.EventSet]'s
        features.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"M":[np.nan, -1., 2.], "N":  [-1, -3, 5]},
            ... )
            >>> a.abs()
            indexes: ...
                    'M': [nan 1. 2.]
                    'N': [1 3 5]
            ...

            ```

        Returns:
            EventSet with positive valued features.
        """
        from temporian.core.operators.unary import abs

        return abs(self)

    def add_index(
        self: EventSetOrNode, indexes: Union[str, List[str]]
    ) -> EventSetOrNode:
        """Adds indexes to an [`EventSet`][temporian.EventSet].

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 1, 0, 1, 1],
            ...     features={
            ...         "f1": [1, 1, 1, 2, 2, 2],
            ...         "f2": [1, 1, 2, 1, 1, 2],
            ...         "f3": [1, 1, 1, 1, 1, 1]
            ...     },
            ... )

            >>> # No index
            >>> a
            indexes: []
            features: [('f1', int64), ('f2', int64), ('f3', int64)]
            events:
                (6 events):
                    timestamps: [0. 1. 1. 1. 1. 2.]
                    'f1': [2 1 1 2 2 1]
                    'f2': [1 1 2 1 2 1]
                    'f3': [1 1 1 1 1 1]
            ...

            >>> # Add only "f1" as index
            >>> b = a.add_index("f1")
            >>> b
            indexes: [('f1', int64)]
            features: [('f2', int64), ('f3', int64)]
            events:
                f1=1 (3 events):
                    timestamps: [1. 1. 2.]
                    'f2': [1 2 1]
                    'f3': [1 1 1]
                f1=2 (3 events):
                    timestamps: [0. 1. 1.]
                    'f2': [1 1 2]
                    'f3': [1 1 1]
            ...

            >>> # Add "f1" and "f2" as indices
            >>> b = a.add_index(["f1", "f2"])
            >>> b
            indexes: [('f1', int64), ('f2', int64)]
            features: [('f3', int64)]
            events:
                f1=1 f2=1 (2 events):
                    timestamps: [1. 2.]
                    'f3': [1 1]
                f1=1 f2=2 (1 events):
                    timestamps: [1.]
                    'f3': [1]
                f1=2 f2=1 (2 events):
                    timestamps: [0. 1.]
                    'f3': [1 1]
                f1=2 f2=2 (1 events):
                    timestamps: [1.]
                    'f3': [1]
            ...

            ```

        Args:
            indexes: List of feature names (strings) that should be added to the
                indexes. These feature names should already exist in the input.

        Returns:
            EventSet with the extended index.

        Raises:
            KeyError: If any of the specified `indexes` are not found in the input.
        """
        from temporian.core.operators.add_index import add_index

        return add_index(self, indexes=indexes)

    def begin(self: EventSetOrNode) -> EventSetOrNode:
        """Generates a single timestamp at the beginning of the
        [`EventSet`][temporian.EventSet], per index group.

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[5, 6, 7, -1],
            ...     features={"f": [50, 60, 70, -10], "idx": [1, 1, 1, 2]},
            ...     indexes=["idx"]
            ... )

            >>> a_ini = a.begin()
            >>> a_ini
            indexes: [('idx', int64)]
            features: []
            events:
                idx=1 (1 events):
                    timestamps: [5.]
                idx=2 (1 events):
                    timestamps: [-1.]
            ...

            ```

        Returns:
            A feature-less EventSet with a single timestamp per index group.
        """
        from temporian.core.operators.begin import begin

        return begin(self)

    def calendar_day_of_month(
        self: EventSetOrNode, tz: Union[str, float, int] = 0
    ) -> EventSetOrNode:
        """Obtains the day of month the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 1 and 31.

        By default, the timezone is UTC unless the `tz` argument is specified,
        as an offset in hours or a timezone name (see `pytz.all_timezones`).

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...    timestamps=["2023-02-04", "2023-02-20", "2023-03-01", "2023-05-07"],
            ... )
            >>> b = a.calendar_day_of_month()
            >>> b
            indexes: ...
            features: [('calendar_day_of_month', int32)]
            events:
                (4 events):
                    timestamps: [...]
                    'calendar_day_of_month': [ 4 20  1  7]
            ...

            ```

        Args:
            tz: timezone name or UTC offset in hours.

        Returns:
            EventSet with a single feature with the day of the month each timestamp
                in `sampling` belongs to.
        """
        from temporian.core.operators.calendar.day_of_month import (
            calendar_day_of_month,
        )

        return calendar_day_of_month(self, tz)

    def calendar_day_of_week(
        self: EventSetOrNode, tz: Union[str, float, int] = 0
    ) -> EventSetOrNode:
        """Obtains the day of the week the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers from 0 (Monday) to 6 (Sunday).

        By default, the timezone is UTC unless the `tz` argument is specified,
        as an offset in hours or a timezone name (see `pytz.all_timezones`).

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...    timestamps=["2023-06-19", "2023-06-21", "2023-06-25", "2023-07-03"],
            ... )
            >>> b = a.calendar_day_of_week()
            >>> b
            indexes: ...
            features: [('calendar_day_of_week', int32)]
            events:
                (4 events):
                    timestamps: [...]
                    'calendar_day_of_week': [0  2  6  0]
            ...

            ```

        Args:
            tz: timezone name or UTC offset in hours.

        Returns:
            EventSet with a single feature with the day of the week each timestamp
                in `sampling` belongs to.
        """
        from temporian.core.operators.calendar.day_of_week import (
            calendar_day_of_week,
        )

        return calendar_day_of_week(self, tz)

    def calendar_hour(
        self: EventSetOrNode, tz: Union[str, float, int] = 0
    ) -> EventSetOrNode:
        """Obtains the hour the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 0 and 23.

        By default, the timezone is UTC unless the `tz` argument is specified,
        as an offset in hours or a timezone name (see `pytz.all_timezones`).

        Basic example with UTC datetimes:
            ```python
            >>> from datetime import datetime
            >>> a = tp.event_set(
            ...    timestamps=[datetime(2020,1,1,18,30), datetime(2020,1,1,23,59)],
            ... )
            >>> b = a.calendar_hour()
            >>> b
            indexes: ...
            features: [('calendar_hour', int32)]
            events:
                (2 events):
                    timestamps: [...]
                    'calendar_hour': [18 23]
            ...

            ```

        Example with timezone:
            ```python
            >>> import pytz
            >>> from datetime import datetime

            >>> # Define a custom timezone (UTC-3)
            >>> custom_tz = pytz.timezone('America/Montevideo')

            >>> # Let's define one event in UTC and another in UTC-3
            >>> a = tp.event_set(
            ...       timestamps=[datetime(2020,1,1,9,00),  # UTC time: 9am
            ...                   datetime(2020,1,1,15,00, tzinfo=custom_tz)
            ...       ]
            ...     )

            >>> # Option 1: specify UTC-3 offset in hours
            >>> a.calendar_hour(tz=-3)
            indexes: ...
                    'calendar_hour': [ 6 15]
            ...

            >>> # Option 2: specify timezone name
            >>> a.calendar_hour(tz="America/Montevideo")
            indexes: ...
                    'calendar_hour': [ 6 15]
            ...

            >>> # No timezone specified, get UTC hour
            >>> a.calendar_hour()
            indexes: ...
                    'calendar_hour': [ 9 18]
            ...



            ```

        Args:
            tz: timezone name or UTC offset in hours.

        Returns:
            EventSet with a single feature with the hour each timestamp in `sampling`
                belongs to.
        """
        from temporian.core.operators.calendar.hour import calendar_hour

        return calendar_hour(self, tz)

    def calendar_iso_week(
        self: EventSetOrNode, tz: Union[str, float, int] = 0
    ) -> EventSetOrNode:
        """Obtains the ISO week the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 1 and 53.

        By default, the timezone is UTC unless the `tz` argument is specified,
        as an offset in hours or a timezone name (see `pytz.all_timezones`).

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...    # Note: 2023-01-01 is Sunday in the same week as 2022-12-31
            ...    timestamps=["2022-12-31", "2023-01-01", "2023-01-02", "2023-12-20"],
            ... )
            >>> b = a.calendar_iso_week()
            >>> b
            indexes: ...
            features: [('calendar_iso_week', int32)]
            events:
                (4 events):
                    timestamps: [...]
                    'calendar_iso_week': [52 52 1 51]
            ...

            ```

        Args:
            tz: timezone name or UTC offset in hours.

        Returns:
            EventSet with a single feature with the ISO week each timestamp in
                `sampling` belongs to.
        """
        from temporian.core.operators.calendar.iso_week import calendar_iso_week

        return calendar_iso_week(self, tz)

    def calendar_day_of_year(
        self: EventSetOrNode, tz: Union[str, float, int] = 0
    ) -> EventSetOrNode:
        """Obtains the day of year the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 1 and 366.

        By default, the timezone is UTC unless the `tz` argument is specified,
        as an offset in hours or a timezone name (see `pytz.all_timezones`).

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...    timestamps=["2020-01-01", "2021-06-01", "2022-12-31", "2024-12-31"],
            ... )
            >>> b = a.calendar_day_of_year()
            >>> b
            indexes: ...
            features: [('calendar_day_of_year', int32)]
            events:
                (4 events):
                    timestamps: [...]
                    'calendar_day_of_year': [ 1 152 365 366]
            ...

            ```

        Args:
            tz: timezone name or UTC offset in hours.

        Returns:
            EventSet with a single feature with the day of the year each timestamp
                in `sampling` belongs to.
        """
        from temporian.core.operators.calendar.day_of_year import (
            calendar_day_of_year,
        )

        return calendar_day_of_year(self, tz)

    def calendar_minute(
        self: EventSetOrNode, tz: Union[str, float, int] = 0
    ) -> EventSetOrNode:
        """Obtain the minute the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between
        0 and 59.

        By default, the timezone is UTC unless the `tz` argument is specified,
        as an offset in hours or a timezone name (see `pytz.all_timezones`).

        Usage example:
            ```python
            >>> from datetime import datetime
            >>> a = tp.event_set(
            ...    timestamps=[datetime(2020,1,1,18,30), datetime(2020,1,1,23,59)],
            ...    name='random_hours'
            ... )
            >>> b = a.calendar_minute()
            >>> b
            indexes: ...
            features: [('calendar_minute', int32)]
            events:
                (2 events):
                    timestamps: [...]
                    'calendar_minute': [30 59]
            ...

            ```

        Args:
            tz: timezone name or UTC offset in hours.

        Returns:
            EventSet with a single feature with the minute each timestamp in
                `sampling` belongs to.
        """
        from temporian.core.operators.calendar.minute import calendar_minute

        return calendar_minute(self, tz)

    def calendar_month(
        self: EventSetOrNode, tz: Union[str, float, int] = 0
    ) -> EventSetOrNode:
        """Obtains the month the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 1 and 12.

        By default, the timezone is UTC unless the `tz` argument is specified,
        as an offset in hours or a timezone name (see `pytz.all_timezones`).

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...    timestamps=["2023-02-04", "2023-02-20", "2023-03-01", "2023-05-07"],
            ...    name='special_events'
            ... )
            >>> b = a.calendar_month()
            >>> b
            indexes: ...
            features: [('calendar_month', int32)]
            events:
                (4 events):
                    timestamps: [...]
                    'calendar_month': [2 2 3 5]
            ...

            ```

        Args:
            tz: timezone name or UTC offset in hours.

        Returns:
            EventSet with a single feature with the month each timestamp in
                `sampling` belongs to.
        """
        from temporian.core.operators.calendar.month import calendar_month

        return calendar_month(self, tz)

    def calendar_second(
        self: EventSetOrNode, tz: Union[str, float, int] = 0
    ) -> EventSetOrNode:
        """Obtains the second the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 0 and 59.

        By default, the timezone is UTC unless the `tz` argument is specified,
        as an offset in hours or a timezone name (see `pytz.all_timezones`).

        Usage example:
            ```python
            >>> from datetime import datetime
            >>> a = tp.event_set(
            ...    timestamps=[datetime(2020,1,1,18,30,55), datetime(2020,1,1,23,59,0)],
            ...    name='random_hours'
            ... )
            >>> b = a.calendar_second()
            >>> b
            indexes: ...
            features: [('calendar_second', int32)]
            events:
                (2 events):
                    timestamps: [...]
                    'calendar_second': [55 0]
            ...

            ```

        Args:
            tz: timezone name or UTC offset in hours.

        Returns:
            EventSet with a single feature with the second each timestamp in
                `sampling` belongs to.
        """
        from temporian.core.operators.calendar.second import calendar_second

        return calendar_second(self, tz)

    def calendar_year(
        self: EventSetOrNode, tz: Union[str, float, int] = 0
    ) -> EventSetOrNode:
        """Obtains the year the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        By default, the timezone is UTC unless the `tz` argument is specified,
        as an offset in hours or a timezone name (see `pytz.all_timezones`).

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...    timestamps=["2021-02-04", "2022-02-20", "2023-03-01", "2023-05-07"],
            ...    name='random_moments'
            ... )
            >>> b = a.calendar_year()
            >>> b
            indexes: ...
            features: [('calendar_year', int32)]
            events:
                (4 events):
                    timestamps: [...]
                    'calendar_year': [2021 2022 2023 2023]
            ...

            ```

        Args:
            tz: timezone name or UTC offset in hours.

        Returns:
            EventSet with a single feature with the year each timestamp in
                `sampling` belongs to.
        """
        from temporian.core.operators.calendar.year import calendar_year

        return calendar_year(self, tz)

    def cast(
        self: EventSetOrNode,
        target: TargetDtypes,
        check_overflow: bool = True,
    ) -> EventSetOrNode:
        """Casts the data types of an [`EventSet`][temporian.EventSet]'s features.

        Features not impacted by cast are kept.

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2],
            ...     features={"A": [0, 2], "B": ['a', 'b'], "C": [5.0, 5.5]},
            ... )

            >>> # Cast all input features to the same dtype
            >>> b = a[["A", "C"]].cast(tp.float32)
            >>> b
            indexes: []
            features: [('A', float32), ('C', float32)]
            events:
                (2 events):
                    timestamps: [1. 2.]
                    'A': [0. 2.]
                    'C': [5.  5.5]
            ...


            >>> # Cast by feature name
            >>> b = a.cast({'A': bool, 'C': int})
            >>> b
            indexes: []
            features: [('A', bool_), ('B', str_), ('C', int64)]
            events:
                (2 events):
                    timestamps: [1. 2.]
                    'A': [False  True]
                    'B': [b'a' b'b']
                    'C': [5  5]
            ...

            >>> # Map original_dtype -> target_dtype
            >>> b = a.cast({float: int, int: float})
            >>> b
            indexes: []
            features: [('A', float64), ('B', str_), ('C', int64)]
            events:
                (2 events):
                    timestamps: [1. 2.]
                    'A': [0. 2.]
                    'B': [b'a' b'b']
                    'C': [5  5]
            ...

            ```

        Args:
            target: Single dtype or a map. Providing a single dtype will cast all
                columns to it. The mapping keys can be either feature names or the
                original dtypes (and not both types mixed), and the values are the
                target dtypes for them. All dtypes must be Temporian types (see
                `dtype.py`).
            check_overflow: Flag to check overflow when casting to a dtype with a
                shorter range (e.g: `INT64`->`INT32`). Note that this check adds
                some computation overhead. Defaults to `True`.

        Returns:
            New EventSet (or the same if no features actually changed dtype),
                with the same feature names as the input one, but with the new
                dtypes as specified in `target`.

        Raises:
            ValueError: If `check_overflow=True` and some value is out of the range
                of the `target` dtype.
            ValueError: If trying to cast a non-numeric string to numeric dtype.
            ValueError: If `target` is not a dtype nor a mapping.
            ValueError: If `target` is a mapping, but some of the keys are not a
                dtype nor a feature in `input.feature_names`, or if those types are
                mixed.
        """
        from temporian.core.operators.cast import cast

        return cast(self, target=target, check_overflow=check_overflow)

    def cumsum(
        self: EventSetOrNode,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the cumulative sum of values over each feature in an
        [`EventSet`][temporian.EventSet].

        Foreach timestamp, calculate the sum of the feature from the beginning.
        Shorthand for `moving_sum(event, window_length=np.inf)`.

        Missing (NaN) values are not accounted for. The output will be NaN until
        the input contains at least one numeric value.

        If `sampling` is specified or `window_length` is an EventSet, the moving
        window is sampled at each timestamp in them, else it is sampled on the
        input's.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 5, 6, 7],
            ...     features={"value": [np.nan, 1, 5, 10, 15, 20]},
            ... )

            >>> b = a.cumsum()
            >>> b
            indexes: ...
                (6 events):
                    timestamps: [0. 1. 2. 5. 6. 7.]
                    'value': [ 0. 1.  6.  16.  31.  51.]
            ...

            ```

        Examples with sampling:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 5, 6, 7],
            ...     features={"value": [np.nan, 1, 5, 10, 15, 20]},
            ... )

            >>> # Cumulative sum at 5 and 10
            >>> b = tp.event_set(timestamps=[5, 10])
            >>> c = a.cumsum(sampling=b)
            >>> c
            indexes: ...
                (2 events):
                    timestamps: [ 5. 10.]
                    'value': [16. 51.]
            ...

            >>> # Sum all values in the EventSet
            >>> c = a.cumsum(sampling=a.end())
            >>> c
            indexes: ...
                (1 events):
                    timestamps: [7.]
                    'value': [51.]
            ...

            ```

        Args:
            sampling: Timestamps to sample the sliding window's value at. If not
                provided, timestamps in the input are used.

        Returns:
            Cumulative sum of each feature.
        """
        from temporian.core.operators.window.moving_sum import cumsum

        return cumsum(self, sampling=sampling)

    def drop_index(
        self: EventSetOrNode,
        indexes: Optional[Union[str, List[str]]] = None,
        keep: bool = True,
    ) -> EventSetOrNode:
        """Removes indexes from an [`EventSet`][temporian.EventSet].

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 1, 0, 1, 1],
            ...     features={
            ...         "f1": [1, 1, 1, 2, 2, 2],
            ...         "f2": [1, 1, 2, 1, 1, 2],
            ...         "f3": [1, 1, 1, 1, 1, 1]
            ...     },
            ...     indexes=["f1", "f2"]
            ... )

            >>> # Both f1 and f2 are indices
            >>> a
            indexes: [('f1', int64), ('f2', int64)]
            features: [('f3', int64)]
            events:
                f1=1 f2=1 (2 events):
                    timestamps: [1. 2.]
                    'f3': [1 1]
                f1=1 f2=2 (1 events):
                    timestamps: [1.]
                    'f3': [1]
                f1=2 f2=1 (2 events):
                    timestamps: [0. 1.]
                    'f3': [1 1]
                f1=2 f2=2 (1 events):
                    timestamps: [1.]
                    'f3': [1]
            ...

            >>> # Drop "f2", remove it from features
            >>> b = a.drop_index("f2", keep=False)
            >>> b
            indexes: [('f1', int64)]
            features: [('f3', int64)]
            events:
                f1=1 (3 events):
                    timestamps: [1. 1. 2.]
                    'f3': [1 1 1]
                f1=2 (3 events):
                    timestamps: [0. 1. 1.]
                    'f3': [1 1 1]
            ...

            >>> # Drop both indices, keep them as features
            >>> b = a.drop_index(["f2", "f1"])
            >>> b
            indexes: []
            features: [('f3', int64), ('f2', int64), ('f1', int64)]
            events:
                (6 events):
                    timestamps: [0. 1. 1. 1. 1. 2.]
                    'f3': [1 1 1 1 1 1]
                    'f2': [2 1 1 2 2 1]
                    'f1': [1 2 1 2 1 1]
            ...

            ```

        Args:
            indexes: Index column(s) to be removed from the input. This can be a
                single column name (`str`) or a list of column names (`List[str]`).
                If not specified or set to `None`, all indexes in the input will
                be removed. Defaults to `None`.
            keep: Flag indicating whether the removed indexes should be kept
                as features in the output EventSet. Defaults to `True`.

        Returns:
            EventSet with the specified indexes removed. If `keep` is set to
            `True`, the removed indexes will be included as features in it.

        Raises:
            ValueError: If an empty list is provided as the `index_names` argument.
            KeyError: If any of the specified `index_names` are missing from
                the input's index.
            ValueError: If a feature name coming from the indexes already exists in
                the input, and the `keep` flag is set to `True`.
        """
        from temporian.core.operators.drop_index import drop_index

        return drop_index(self, indexes=indexes, keep=keep)

    def end(self: EventSetOrNode) -> EventSetOrNode:
        """Generates a single timestamp at the end of an
        [`EventSet`][temporian.EventSet], per index key.

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[5, 6, 7, 1],
            ...     features={"f": [50, 60, 70, 10], "idx": [1, 1, 1, 2]},
            ...     indexes=["idx"]
            ... )

            >>> a_end = a.end()
            >>> a_end
            indexes: [('idx', int64)]
            features: []
            events:
                idx=1 (1 events):
                    timestamps: [7.]
                idx=2 (1 events):
                    timestamps: [1.]
            ...

            ```

        Returns:
            A feature-less EventSet with a single timestamp per index group.
        """
        from temporian.core.operators.end import end

        return end(self)

    def enumerate(self: EventSetOrNode) -> EventSetOrNode:
        """Create an `int64` feature with the ordinal position of each event in an
        [`EventSet`][temporian.EventSet].

        Each index group is enumerated independently.

        Usage:
            ```python
            >>> a = tp.event_set(
            ...    timestamps=[-1, 2, 3, 5, 0],
            ...    features={"cat": ["A", "A", "A", "A", "B"]},
            ...    indexes=["cat"],
            ... )
            >>> b = a.enumerate()
            >>> b
            indexes: [('cat', str_)]
            features: [('enumerate', int64)]
            events:
                cat=b'A' (4 events):
                    timestamps: [-1.  2.  3.  5.]
                    'enumerate': [0 1 2 3]
                cat=b'B' (1 events):
                    timestamps: [0.]
                    'enumerate': [0]
            ...

            ```

        Returns:
            EventSet with a single feature with each event's ordinal position in
                its index group.
        """
        from temporian.core.operators.enumerate import enumerate

        return enumerate(self)

    def equal(self: EventSetOrNode, other: Any) -> EventSetOrNode:
        """Checks element-wise equality of an [`EventSet`][temporian.EventSet]
        to another one or to a single value.

        Each feature is compared element-wise to the feature in
        `other` in the same position.
        Note that it will always return False on NaN elements.

        Inputs must have the same sampling and the same number of features.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f1": [0, 100, 200]}
            ... )
            >>> b = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"f2": [-10, 100, 5]},
            ...     same_sampling_as=a
            ... )

            >>> # WARN: Don't use this for element-wise comparison
            >>> a == b
            False

            >>> # Element-wise comparison to a scalar value
            >>> c = a.equal(100)
            >>> c
            indexes: []
            features: [('f1', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'f1': [False True False]
            ...

            >>> # Element-wise comparison between two EventSets
            >>> c = a.equal(b)
            >>> c
            indexes: []
            features: [('eq_f1_f2', bool_)]
            events:
                (3 events):
                    timestamps: [1. 2. 3.]
                    'eq_f1_f2': [False True False]
            ...

            ```

        Args:
            other: Second EventSet or single value to compare.

        Returns:
            EventSet with boolean features.
        """
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import equal

            return equal(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR + (bool, str)):
            from temporian.core.operators.scalar import equal_scalar

            return equal_scalar(input=self, value=other)

        self._raise_error("equal", other, "int,float,bool,str")
        assert False

    def experimental_fast_fourier_transform(
        self: EventSetOrNode,
        *,
        num_events: int,
        hop_size: Optional[int] = None,
        window: Optional[str] = None,
        num_spectral_lines: Optional[int] = None,
    ) -> EventSetOrNode:
        """Computes the Fast Fourier Transform of an
        [`EventSet`][temporian.EventSet] with a single tp.float32 feature.

        WARNING: This operator is experimental. The implementation is not yet
        optimized for speed, and the operator signature might change in the
        future.

        The window length is defined in number of events, instead of
        timestamp duration like most other operators. The 'num_events' argument
        needs to be specified by warg i.e. fast_fourier_transform(num_events=5)
        instead of fast_fourier_transform(5).

        The operator returns the amplitude of each spectral line as
        separate tp.float32 features named "a0", "a1", "a2", etc. By default,
        `num_events // 2` spectral line are returned.

        Usage:
            ```python
            >>> a = tp.event_set(
            ...    timestamps=[1,2,3,4,5,6],
            ...    features={"x": [4.,3.,2.,6.,2.,1.]},
            ... )
            >>> b = a.experimental_fast_fourier_transform(num_events=4, window="hamming")
            >>> b
            indexes: []
            features: [('a0', float64), ('a1', float64)]
            events:
                 (2 events):
                    timestamps: [4. 6.]
                    'a0': [4.65 6.4 ]
                    'a1': [2.1994 4.7451]
            ...

            ```

        Args:
            num_events: Size of the FFT expressed as a number of events.
            window: Optional window function applied before the FFT. if None, no
                window is applied. Supported values are: "hamming".
            hop_size: Step, in number of events, between consecutive outputs.
                Default to num_events//2.
            num_spectral_lines: Number of returned spectral lines. If set, the
                operators returns the `num_spectral_lines` low frequency
                spectral lines. `num_spectral_lines` should be between 1 and
                `num_events`.

        Returns:
            EventSet containing the amplitude of each frequency band of the
                Fourier Transform.
        """
        from temporian.core.operators.fast_fourier_transform import (
            fast_fourier_transform,
        )

        return fast_fourier_transform(
            self,
            num_events=num_events,
            hop_size=hop_size,
            window=window,
            num_spectral_lines=num_spectral_lines,
        )

    def filter(
        self: EventSetOrNode,
        condition: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Filters out events in an [`EventSet`][temporian.EventSet] for which a
        condition is false.

        Each timestamp in the input is only kept if the corresponding value for that
        timestamp in `condition` is `True`.

        the input and `condition` must have the same sampling, and `condition` must
        have one single feature, of boolean type.

        filter(x) is equivalent to filter(x,x). filter(x) can be used to convert
        a boolean mask into a timestamps.

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 5, 6],
            ...     features={"f1": [0, 10, 50, 60], "f2": [50, 100, 500, 600]},
            ... )

            >>> # Example boolean condition
            >>> condition = a["f1"] > 20
            >>> condition
            indexes: ...
                    timestamps: [0. 1. 5. 6.]
                    'f1': [False False  True  True]
            ...

            >>> # Filter only True timestamps
            >>> filtered = a.filter(condition)
            >>> filtered
            indexes: ...
                    timestamps: [5. 6.]
                    'f1': [50 60]
                    'f2': [500 600]
            ...

            ```

        Args:
            condition: EventSet with a single boolean feature.

        Returns:
            Filtered EventSet.
        """
        from temporian.core.operators.filter import filter

        return filter(self, condition=condition)

    def isnan(
        self: EventSetOrNode,
    ) -> EventSetOrNode:
        """Returns boolean features, `True` in the NaN elements of the
        [`EventSet`][temporian.EventSet].

        Note that for `int` and `bool` this will always be `False` since those types
        don't support NaNs. It only makes actual sense to use on `float` (or
        `tp.float32`) features.

        See also `evset.notnan()`.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"M":[np.nan, 5., np.nan], "N":  [-1, 0, 5]},
            ... )
            >>> b = a.isnan()
            >>> b
            indexes: ...
                    'M': [ True False True]
                    'N': [False False False]
            ...

            >>> # Count nans
            >>> b["M"].cast(int).cumsum()
            indexes: ...
                    timestamps: [1. 2. 3.]
                    'M': [1 1 2]
            ...

            ```

        Returns:
            EventSet with boolean features.
        """
        from temporian.core.operators.unary import isnan

        return isnan(self)

    def join(
        self: EventSetOrNode,
        other: EventSetOrNode,
        how: str = "left",
        on: Optional[str] = None,
    ) -> EventSetOrNode:
        """Join [`EventSets`][temporian.EventSet] with different samplings.

        Join features from two EventSets based on timestamps. Optionally, join on
        timestamps and an extra `int64` feature. Joined EventSets should have the
        same index and non-overlapping feature names.

        To concatenate EventSets with the same sampling, use
        [`tp.glue()`][temporian.glue] instead. [`tp.glue()`][temporian.glue] is
        almost free while [`EventSet.join()`][temporian.EventSet.join] can be expensive.

        To resample an EventSets according to another EventSets's sampling, use
        [`EventSet.resample()`][temporian.EventSet.resample] instead.

        Example:

            ```python
            >>> a = tp.event_set(timestamps=[0, 1, 2], features={"A": [0, 10, 20]})
            >>> b = tp.event_set(timestamps=[0, 2, 4], features={"B": [0., 2., 4.]})

            >>> # Left join
            >>> c = a.join(b)
            >>> c
            indexes: []
            features: [('A', int64), ('B', float64)]
            events:
                (3 events):
                    timestamps: [0. 1. 2.]
                    'A': [ 0 10 20]
                    'B': [ 0. nan 2.]
            ...

            ```

        Example with an index and feature join:

            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 1, 1],
            ...     features={
            ...         "idx": [1, 1, 2, 2],
            ...         "match": [1, 2, 4, 5],
            ...         "A": [10, 20, 40, 50],
            ...     },
            ...     indexes=["idx"]
            ... )
            >>> b = tp.event_set(
            ...     timestamps=[0, 1, 0, 1, 1, 1],
            ...     features={
            ...         "idx": [1, 1, 2, 2, 2, 2],
            ...         "match": [1, 2, 3, 4, 5, 6],
            ...         "B": [10., 20., 30., 40., 50., 60.],
            ...     },
            ...     indexes=["idx"]
            ... )

            >>> # Join by index and 'match'
            >>> c = a.join(b, on="match")
            >>> c
            indexes: [('idx', int64)]
            features: [('match', int64), ('A', int64), ('B', float64)]
            events:
                idx=1 (2 events):
                    timestamps: [0. 1.]
                    'match': [1 2]
                    'A': [10 20]
                    'B': [10. 20.]
                idx=2 (2 events):
                    timestamps: [1. 1.]
                    'match': [4 5]
                    'A': [40 50]
                    'B': [40. 50.]
            ...

            ```

        Args:
            other: Right EventSet to join.
            how: Whether to perform a `"left"`, `"inner"`, or `"outer"` join.
                Currently, only `"left"` join is supported.
            on: Optional extra int64 feature name to join on.

        Returns:
            The joined EventSets.
        """
        from temporian.core.operators.join import join

        return join(left=self, right=other, how=how, on=on)

    def lag(self: EventSetOrNode, duration: Duration) -> EventSetOrNode:
        """Adds a delay to an [`EventSet`][temporian.EventSet]'s timestamps.

        In other words, shifts the timestamp values forwards in time.

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 5, 6],
            ...     features={"value": [0, 1, 5, 6]},
            ... )

            >>> b = a.lag(tp.duration.seconds(2))
            >>> b
            indexes: ...
                (4 events):
                    timestamps: [2. 3. 7. 8.]
                    'value': [0 1 5 6]
            ...

            ```

        Args:
            duration: Duration to lag by.

        Returns:
            Lagged EventSet.
        """
        from temporian.core.operators.lag import lag

        return lag(self, duration=duration)

    def leak(self: EventSetOrNode, duration: Duration) -> EventSetOrNode:
        """Subtracts a duration from an [`EventSet`][temporian.EventSet]'s
        timestamps.

        In other words, shifts the timestamp values backward in time.

        Note that this operator moves future data into the past, and should be used
        with caution to prevent unwanted future leakage. For instance, this op
        should generally not be used to compute the input features of a model.

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 5, 6],
            ...     features={"value": [0, 1, 5, 6]},
            ... )

            >>> b = a.leak(tp.duration.seconds(2))
            >>> b
            indexes: ...
                (4 events):
                    timestamps: [-2. -1. 3. 4.]
                    'value': [0 1 5 6]
            ...

            ```

        Args:
            duration: Duration to leak by.

        Returns:
            Leaked EventSet.
        """
        from temporian.core.operators.leak import leak

        return leak(self, duration=duration)

    def map(
        self: EventSetOrNode,
        func: MapFunction,
        output_dtypes: Optional[TargetDtypes] = None,
        receive_extras: bool = False,
    ) -> EventSetOrNode:
        """Applies a function on each value of an
        [`EventSet`][temporian.EventSet]'s features.

        The function receives the scalar value, and if `receive_extras` is True,
        also a [`MapExtras`][temporian.types.MapExtras] object containing
        information about the value's position in the EventSet. The MapExtras
        object should not be modified by the function, since it is shared across
        all calls.

        If the output of the functon has a different dtype than the input, the
        `output_dtypes` argument must be specified.

        This operator is slow. When possible, existing operators should be used.

        A Temporian graph with a `map` operator is not serializable.

        Usage example with lambda function:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2],
            ...     features={"value": [10, 20, 30]},
            ... )

            >>> b = a.map(lambda v: v + 1)
            >>> b
            indexes: ...
                (3 events):
                    timestamps: [0. 1. 2.]
                    'value': [11 21 31]
            ...

            ```

        Usage example with `output_dtypes`:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2],
            ...     features={"a": [10, 20, 30], "b": ["100", "200", "300"]},
            ... )

            >>> def f(value):
            ...     if value.dtype == np.int64:
            ...         return float(value) + 1
            ...     else:
            ...         return int(value) + 2

            >>> b = a.map(f, output_dtypes={"a": float, "b": int})
            >>> b
            indexes: ...
                (3 events):
                    timestamps: [0. 1. 2.]
                    'a': [11. 21. 31.]
                    'b': [102 202 302]
            ...

            ```

        Usage example with `MapExtras`:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2],
            ...     features={"value": [10, 20, 30]},
            ... )

            >>> def f(value, extras):
            ...     return f"{extras.feature_name}-{extras.timestamp}-{value}"

            >>> b = a.map(f, output_dtypes=str, receive_extras=True)
            >>> b
            indexes: ...
                (3 events):
                    timestamps: [0. 1. 2.]
                    'value': [b'value-0.0-10' b'value-1.0-20' b'value-2.0-30']
            ...

            ```

        Args:
            func: The function to apply on each value.
            output_dtypes: Expected dtypes of the output feature(s) after
                applying the function to them. If not provided, the output
                dtypes will be expected to be the same as the input ones. If a
                single dtype, all features will be expected to have that dtype.
                If a mapping, the keys can be either feature names or the
                input dtypes (and not both types mixed), and the values are the
                target dtypes for them. All dtypes must be Temporian types (see
                `dtype.py`).
            receive_extras: Whether the function should receive a
                [`MapExtras`][temporian.types.MapExtras] object as second
                argument.

        Returns:
            EventSet with the function applied on each value.
        """
        from temporian.core.operators.map import map as tp_map

        return tp_map(
            self,
            func=func,
            output_dtypes=output_dtypes,
            receive_extras=receive_extras,
        )

    def log(self: EventSetOrNode) -> EventSetOrNode:
        """Calculates the natural logarithm of an [`EventSet`][temporian.EventSet]'s
        features.

        Can only be used on floating point features.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3, 4, 5],
            ...     features={"M": [np.e, 1., 2., 10., -1.]},
            ... )
            >>> a.log()
            indexes: ...
                    timestamps: [1. 2. 3. 4. 5.]
                    'M': [1. 0. 0.6931 2.3026 nan]
            ...

            ```

        Returns:
            EventSetOr with logarithm of input features.
        """
        from temporian.core.operators.unary import log

        return log(self)

    def moving_count(
        self: EventSetOrNode,
        window_length: WindowLength,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Gets the number of events in a sliding window.

        Create a tp.int32 feature containing the number of events in the time
        window (t - window_length, t].

        `sampling` can't be  specified if a variable `window_length` is
        specified (i.e. if `window_length` is an EventSet).

        If `sampling` is specified or `window_length` is an EventSet, the moving
        window is sampled at each timestamp in them, else it is sampled on the
        input's.

        Example without sampling:
            ```python
            >>> a = tp.event_set(timestamps=[0, 1, 2, 5, 6, 7])
            >>> b = a.moving_count(tp.duration.seconds(2))
            >>> b
            indexes: ...
                (6 events):
                    timestamps: [0. 1. 2. 5. 6. 7.]
                    'count': [1 2 2 1 2 2]
            ...

            ```

        Example with sampling:
            ```python
            >>> a = tp.event_set(timestamps=[0, 1, 2, 5])
            >>> b = tp.event_set(timestamps=[-1, 0, 1, 2, 3, 4, 5, 6, 7])
            >>> c = a.moving_count(tp.duration.seconds(2), sampling=b)
            >>> c
            indexes: ...
                (9 events):
                    timestamps: [-1. 0. 1. 2. 3. 4. 5. 6. 7.]
                    'count': [0 1 2 2 1 0 1 1 0]
            ...

            ```

        Example with variable window length:
            ```python
            >>> a = tp.event_set(timestamps=[0, 1, 2, 5])
            >>> b = tp.event_set(
            ...     timestamps=[0, 3, 3, 3, 9],
            ...     features={
            ...         "w": [1, 0.5, 3.5, 2.5, 5],
            ...     },
            ... )
            >>> c = a.moving_count(window_length=b)
            >>> c
            indexes: []
            features: [('count', int32)]
            events:
                (5 events):
                    timestamps: [0. 3. 3. 3. 9.]
                    'count': [1 0 3 2 1]
            ...

            ```

        Example with index:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3, 0, 1, 2],
            ...     features={
            ...         "idx": ["i1", "i1", "i1", "i2", "i2", "i2"],
            ...     },
            ...     indexes=["idx"],
            ... )
            >>> b = a.moving_count(tp.duration.seconds(2))
            >>> b
            indexes: [('idx', str_)]
            features: [('count', int32)]
            events:
                idx=b'i1' (3 events):
                    timestamps: [1. 2. 3.]
                    'count': [1 2 2]
                idx=b'i2' (3 events):
                    timestamps: [0. 1. 2.]
                    'count': [1 2 2]
            ...

            ```

        Args:
            window_length: Sliding window's length.
            sampling: Timestamps to sample the sliding window's value at. If not
                provided, timestamps in `input` are used.

        Returns:
            EventSet containing the count of events in `input` in a moving
                window.
        """
        from temporian.core.operators.window.moving_count import moving_count

        return moving_count(
            self, window_length=window_length, sampling=sampling
        )

    def moving_max(
        self: EventSetOrNode,
        window_length: WindowLength,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the maximum in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each index and feature independently,
        returns at time t the max of non-nan values for the feature in the window
        (t - window_length, t].

        `sampling` can't be  specified if a variable `window_length` is
        specified (i.e. if `window_length` is an EventSet).

        If `sampling` is specified or `window_length` is an EventSet, the moving
        window is sampled at each timestamp in them, else it is sampled on the
        input's.

        If the window does not contain any values (e.g., all the values are
        missing, or the window does not contain any sampling), outputs missing
        values.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 5, 6, 7],
            ...     features={"value": [np.nan, 1, 5, 1, 15, 20]},
            ... )

            >>> b = a.moving_max(tp.duration.seconds(4))
            >>> b
            indexes: ...
                (6 events):
                    timestamps: [0. 1. 2. 5. 6. 7.]
                    'value': [nan 1. 5. 5. 15. 20.]
            ...

            ```

        See [`EventSet.moving_count()`][temporian.EventSet.moving_count] for
        examples with external sampling and indices.

        Args:
            window_length: Sliding window's length.
            sampling: Timestamps to sample the sliding window's value at. If not
                provided, timestamps in the input are used.

        Returns:
            EventSet containing the max of each feature in the input.
        """
        from temporian.core.operators.window.moving_max import moving_max

        return moving_max(self, window_length=window_length, sampling=sampling)

    def moving_min(
        self: EventSetOrNode,
        window_length: WindowLength,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the minimum of values in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each index and feature independently,
        returns at time t the minimum of non-nan values for the feature in the window
        (t - window_length, t].

        `sampling` can't be  specified if a variable `window_length` is
        specified (i.e. if `window_length` is an EventSet).

        If `sampling` is specified or `window_length` is an EventSet, the moving
        window is sampled at each timestamp in them, else it is sampled on the
        input's.

        If the window does not contain any values (e.g., all the values are
        missing, or the window does not contain any sampling), outputs missing
        values.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 5, 6, 7],
            ...     features={"value": [np.nan, 1, 5, 10, 15, 20]},
            ... )

            >>> b = a.moving_min(tp.duration.seconds(4))
            >>> b
            indexes: ...
                (6 events):
                    timestamps: [0. 1. 2. 5. 6. 7.]
                    'value': [nan 1. 1. 5. 10. 10.]
            ...

            ```

        See [`EventSet.moving_count()`][temporian.EventSet.moving_count] for
        examples of moving window operations with external sampling and indices.

        Args:
            window_length: Sliding window's length.
            sampling: Timestamps to sample the sliding window's value at. If not
                provided, timestamps in the input are used.

        Returns:
            EventSet containing the minimum of each feature in the input.
        """
        from temporian.core.operators.window.moving_min import moving_min

        return moving_min(self, window_length=window_length, sampling=sampling)

    def moving_standard_deviation(
        self: EventSetOrNode,
        window_length: WindowLength,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the standard deviation of values in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each feature independently, returns at
        time t the standard deviation for the feature in the window
        (t - window_length, t].

        `sampling` can't be  specified if a variable `window_length` is
        specified (i.e. if `window_length` is an EventSet).

        If `sampling` is specified or `window_length` is an EventSet, the moving
        window is sampled at each timestamp in them, else it is sampled on the
        input's.

        Missing values (such as NaNs) are ignored.

        If the window does not contain any values (e.g., all the values are
        missing, or the window does not contain any sampling), outputs missing
        values.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 5, 6, 7],
            ...     features={"value": [np.nan, 1, 5, 10, 15, 20]},
            ... )

            >>> b = a.moving_standard_deviation(tp.duration.seconds(4))
            >>> b
            indexes: ...
                (6 events):
                    timestamps: [0. 1. 2. 5. 6. 7.]
                    'value': [ nan 0.  2.  2.5  2.5  4.0825]
            ...

            ```

        See [`EventSet.moving_count()`][temporian.EventSet.moving_count] for
        examples of moving window operations with external sampling and indices.

        Args:
            window_length: Sliding window's length.
            sampling: Timestamps to sample the sliding window's value at. If not
                provided, timestamps in the input are used.

        Returns:
            EventSet containing the moving standard deviation of each feature in
                the input.
        """
        from temporian.core.operators.window.moving_standard_deviation import (
            moving_standard_deviation,
        )

        return moving_standard_deviation(
            self, window_length=window_length, sampling=sampling
        )

    def moving_sum(
        self: EventSetOrNode,
        window_length: WindowLength,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the sum of values in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each feature independently, returns at
        time t the sum of values for the feature in the window
        (t - window_length, t].

        `sampling` can't be  specified if a variable `window_length` is
        specified (i.e. if `window_length` is an EventSet).

        If `sampling` is specified or `window_length` is an EventSet, the moving
        window is sampled at each timestamp in them, else it is sampled on the
        input's.

        Missing values (such as NaNs) are ignored.

        If the window does not contain any values (e.g., all the values are
        missing, or the window does not contain any sampling), outputs missing
        values.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 5, 6, 7],
            ...     features={"value": [np.nan, 1, 5, 10, 15, 20]},
            ... )

            >>> b = a.moving_sum(tp.duration.seconds(4))
            >>> b
            indexes: ...
                (6 events):
                    timestamps: [0. 1. 2. 5. 6. 7.]
                    'value': [ 0. 1.  6.  15.  25.  45.]
            ...

            ```

        See [`EventSet.moving_count()`][temporian.EventSet.moving_count] for
        examples of moving window operations with external sampling and indices.

        Args:
            window_length: Sliding window's length.
            sampling: Timestamps to sample the sliding window's value at. If not
                provided, timestamps in the input are used.

        Returns:
            EventSet containing the moving sum of each feature in the input.
        """
        from temporian.core.operators.window.moving_sum import moving_sum

        return moving_sum(self, window_length=window_length, sampling=sampling)

    def notnan(
        self: EventSetOrNode,
    ) -> EventSetOrNode:
        """Returns boolean features, `False` in the NaN elements of an
        [`EventSet`][temporian.EventSet].

        Equivalent to `~evset.isnan(...)`.

        Note that for `int` and `bool` this will always be `True` since those types
        don't support NaNs. It only makes actual sense to use on `float` (or
        `tp.float32`) features.

        See also `evset.isnan()`.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 3],
            ...     features={"M":[np.nan, 5., np.nan], "N":  [-1, 0, 5]},
            ... )
            >>> b = a.notnan()
            >>> b
            indexes: ...
                    'M': [False True False]
                    'N': [ True True True]
            ...

            >>> # Filter only rows where "M" is not nan
            >>> a.filter(b["M"])
            indexes: ...
                    'M': [5.]
                    'N': [0]
            ...

            ```

        Returns:
            EventSet with boolean features.
        """
        from temporian.core.operators.unary import notnan

        return notnan(self)

    def prefix(
        self: EventSetOrNode,
        prefix: str,
    ) -> EventSetOrNode:
        """Adds a prefix to the names of the features in an
        [`EventSet`][temporian.EventSet].

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...    timestamps=[0, 1],
            ...    features={"f1": [0, 2], "f2": [5, 6]}
            ... )
            >>> b = a * 5

            >>> # Prefix before glue to avoid duplicated names
            >>> c = tp.glue(a.prefix("original_"), b.prefix("result_"))
            >>> c
            indexes: ...
                    'original_f1': [0 2]
                    'original_f2': [5 6]
                    'result_f1': [ 0 10]
                    'result_f2': [25 30]
            ...

            ```

        Args:
            prefix: Prefix to add in front of the feature names.

        Returns:
            Prefixed EventSet.
        """
        from temporian.core.operators.prefix import prefix as _prefix

        return _prefix(self, prefix=prefix)

    def propagate(
        self: EventSetOrNode, sampling: EventSetOrNode, resample: bool = False
    ) -> EventSetOrNode:
        """Propagates feature values over another [`EventSet`][temporian.EventSet]'s
        index.

        Given the input and `sampling` where the input's indexes are a subset of
        `sampling`'s (e.g., the indexes of the input are `["x"]`, and the indexes of
        `sampling` are `["x","y"]`), duplicates the features of the input over the
        indexes of `sampling`.

        Example use case:
            ```python
            >>> products = tp.event_set(
            ...     timestamps=[1, 2, 3, 1, 2, 3],
            ...     features={
            ...         "product": [1, 1, 1, 2, 2, 2],
            ...         "sales": [100., 200., 500., 1000., 2000., 5000.]
            ...     },
            ...     indexes=["product"],
            ... )
            >>> store = tp.event_set(
            ...     timestamps=[1, 2, 3, 4, 5],
            ...     features={
            ...         "sales": [10000., 20000., 30000., 5000., 1000.]
            ...     },
            ... )

            >>> # First attempt: divide to calculate fraction of total store sales
            >>> products / store
            Traceback (most recent call last):
                ...
            ValueError: Arguments don't have the same index. ...

            >>> # Second attempt: propagate index
            >>> store_prop = store.propagate(products)
            >>> products / store_prop
            Traceback (most recent call last):
                ...
            ValueError: Arguments should have the same sampling. ...

            >>> # Third attempt: propagate + resample
            >>> store_resample = store.propagate(products, resample=True)
            >>> div = products / store_resample
            >>> div
            indexes: [('product', int64)]
            features: [('div_sales_sales', float64)]
            events:
                product=1 (3 events):
                    timestamps: [1. 2. 3.]
                    'div_sales_sales': [0.01   0.01   0.0167]
                product=2 (3 events):
                    timestamps: [1. 2. 3.]
                    'div_sales_sales': [0.1    0.1    0.1667]
            ...

            ```

        Args:
            sampling: EventSet with the indexes to propagate to.
            resample: If true, apply a [`EventSet.resample()`][temporian.EventSet.resample]
                before propagating, for the output to have the same sampling as
                `sampling`.

        Returns:
            EventSet propagated over `sampling`'s index.
        """
        from temporian.core.operators.propagate import propagate

        return propagate(self, sampling=sampling, resample=resample)

    def rename(
        self: EventSetOrNode,
        features: Optional[Union[str, Dict[str, str]]] = None,
        indexes: Optional[Union[str, Dict[str, str]]] = None,
    ) -> EventSetOrNode:
        """Renames an [`EventSet`][temporian.EventSet]'s features and index.

        If the input has a single feature, then the `features` can be a
        single string with the new name.

        If the input has multiple features, then `features` must be a mapping
        with the old names as keys and the new names as values.

        The indexes renaming follows the same criteria, accepting a single string or
        a mapping for multiple indexes.

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...    timestamps=[0, 1],
            ...    features={"f1": [0, 2], "f2": [5, 6]}
            ... )
            >>> b = 5 * a

            >>> # Rename single feature
            >>> b_1 = b["f1"].rename("f1_result")
            >>> b_1
            indexes: []
            features: [('f1_result', int64)]
            events:
                (2 events):
                    timestamps: [0. 1.]
                    'f1_result': [ 0 10]
            ...

            >>> # Rename multiple features
            >>> b_rename = b.rename({"f1": "5xf1", "f2": "5xf2"})
            >>> b_rename
            indexes: []
            features: [('5xf1', int64), ('5xf2', int64)]
            events:
                (2 events):
                    timestamps: [0. 1.]
                    '5xf1': [ 0 10]
                    '5xf2': [25 30]
            ...

            ```

        Args:
            features: New feature name or mapping from old names to new names.
            indexes: New index name or mapping from old names to new names.

        Returns:
            EventSet with renamed features and index.
        """
        from temporian.core.operators.rename import rename

        return rename(self, features=features, indexes=indexes)

    def resample(
        self: EventSetOrNode,
        sampling: EventSetOrNode,
    ) -> EventSetOrNode:
        """Resamples an [`EventSet`][temporian.EventSet] at each timestamp of
        another [`EventSet`][temporian.EventSet].

        If a timestamp in `sampling` does not have a corresponding timestamp in
        the input, the last timestamp in the input is used instead. If this timestamp
        is anterior to an value in the input, the value is replaced by
        `dtype.MissingValue(...)`.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 5, 8, 9],
            ...     features={"f1": [1.0, 2.0, 3.0, 4.0]}
            ... )
            >>> b = tp.event_set(timestamps=[-1, 1, 6, 10])
            >>> c = a.resample(b)
            >>> c
            indexes: ...
                    timestamps: [-1.  1.  6. 10.]
                    'f1': [nan  1.  2.  4.]
            ...

            ```

        Args:
            sampling: EventSet to use the sampling of.

        Returns:
            Resampled EventSet, with same sampling as `sampling`.
        """
        from temporian.core.operators.resample import resample

        return resample(self, sampling=sampling)

    def select(
        self: EventSetOrNode,
        feature_names: Union[str, List[str]],
    ) -> EventSetOrNode:
        """Selects a subset of features from an [`EventSet`][temporian.EventSet].

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2],
            ...     features={"A": [1, 2], "B": ['s', 'm'], "C": [5.0, 5.5]},
            ... )

            >>> # Select single feature
            >>> b = a.select('B')
            >>> # Equivalent
            >>> b = a['B']
            >>> b
            indexes: []
            features: [('B', str_)]
            events:
                (2 events):
                    timestamps: [1. 2.]
                    'B': [b's' b'm']
            ...

            >>> # Select multiple features
            >>> bc = a.select(['B', 'C'])
            >>> # Equivalent
            >>> bc = a[['B', 'C']]
            >>> bc
            indexes: []
            features: [('B', str_), ('C', float64)]
            events:
                (2 events):
                    timestamps: [1. 2.]
                    'B': [b's' b'm']
                    'C': [5.  5.5]
            ...

            ```

        Args:
            feature_names: Name or list of names of the features to select from the
                input.

        Returns:
            EventSet containing only the selected features.
        """
        from temporian.core.operators.select import select

        return select(self, feature_names=feature_names)

    def select_index_values(
        self: EventSetOrNode,
        keys: Optional[IndexKeyList] = None,
        *,
        number: Optional[int] = None,
        fraction: Optional[float] = None,
    ) -> EventSetOrNode:
        """Selects a subset of index values from an
        [`EventSet`][temporian.EventSet].

        Exactly one of `keys`, `number`, or `fraction` should be provided.

        If `number` or `fraction` is specified, the index values are selected
        randomly.

        If `fraction` is specified and `fraction * len(index keys)` doesn't
        result in an integer, the number of index values selected is rounded
        down.

        If used in compiled or graph mode, the specified keys are compiled as-is
        along with the operator, which means that they must be available when
        loading and running the graph on new data.

        Example with `keys` with a single index and a single key:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 3],
            ...     features={
            ...         "f": [10, 20, 30, 40],
            ...         "x": ["A", "B", "A", "B"],
            ...     },
            ...     indexes=["x"],
            ... )
            >>> b = a.select_index_values("A")
            >>> b
            indexes: [('x', str_)]
            features: [('f', int64)]
            events:
                x=b'A' (2 events):
                    timestamps: [0. 2.]
                    'f': [10 30]
            ...

            ```

        Example with `keys` with multiple indexes and keys:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 3],
            ...     features={
            ...         "f": [10, 20, 30, 40],
            ...         "x": [1, 1, 2, 2],
            ...         "y": ["A", "B", "A", "B"],
            ...     },
            ...     indexes=["x", "y"],
            ... )
            >>> b = a.select_index_values([(1, "A"), (2, "B")])
            >>> b
            indexes: [('x', int64), ('y', str_)]
            features: [('f', int64)]
            events:
                x=1 y=b'A' (1 events):
                    timestamps: [0.]
                    'f': [10]
                x=2 y=b'B' (1 events):
                    timestamps: [3.]
                    'f': [40]
            ...

            ```

        Example with `number`:
            ```python
            >>> import random
            >>> random.seed(0)

            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 3],
            ...     features={
            ...         "f": [10, 20, 30, 40],
            ...         "x": [1, 1, 2, 2],
            ...         "y": ["A", "B", "A", "B"],
            ...     },
            ...     indexes=["x", "y"],
            ... )
            >>> b = a.select_index_values(number=2)
            >>> b
            indexes: [('x', int64), ('y', str_)]
            features: [('f', int64)]
            events:
                x=1 y=b'A' (1 events):
                    timestamps: [0.]
                    'f': [10]
                x=2 y=b'A' (1 events):
                    timestamps: [2.]
                    'f': [30]
            ...

            ```

        Example with `fraction`:
            ```python
            >>> import random
            >>> random.seed(0)

            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 3],
            ...     features={
            ...         "f": [10, 20, 30, 40],
            ...         "x": [1, 1, 2, 2],
            ...         "y": ["A", "B", "A", "B"],
            ...     },
            ...     indexes=["x", "y"],
            ... )
            >>> b = a.select_index_values(fraction=0.75)
            >>> b
            indexes: [('x', int64), ('y', str_)]
            features: [('f', int64)]
            events:
                x=1 y=b'A' (1 events):
                    timestamps: [0.]
                    'f': [10]
                x=2 y=b'A' (1 events):
                    timestamps: [2.]
                    'f': [30]
            ...

            ```

        Args:
            keys: index key or list of index keys to select from the EventSet.
            number: number of index values to select. If `number` is greater
                than the number of index values, all the index values are
                selected.
            fraction: fraction of index values to select, expressed as a float
                between 0 and 1.

        Returns:
            EventSet with a subset of the index values.
        """
        from temporian.core.operators.select_index_values import (
            select_index_values,
        )

        return select_index_values(
            self, keys=keys, number=number, fraction=fraction
        )

    def set_index(
        self: EventSetOrNode, indexes: Union[str, List[str]]
    ) -> EventSetOrNode:
        """Replaces the index in an [`EventSet`][temporian.EventSet].

        Usage example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[1, 2, 1, 0, 1, 1],
            ...     features={
            ...         "f1": [1, 1, 1, 2, 2, 2],
            ...         "f2": [1, 1, 2, 1, 1, 2],
            ...         "f3": [1, 1, 1, 1, 1, 1]
            ...     },
            ...     indexes=["f1"],
            ... )

            >>> # "f1" is the current index
            >>> a
            indexes: [('f1', int64)]
            features: [('f2', int64), ('f3', int64)]
            events:
                f1=1 (3 events):
                    timestamps: [1. 1. 2.]
                    'f2': [1 2 1]
                    'f3': [1 1 1]
                f1=2 (3 events):
                    timestamps: [0. 1. 1.]
                    'f2': [1 1 2]
                    'f3': [1 1 1]
            ...

            >>> # Set "f2" as the only index, remove "f1"
            >>> b = a.set_index("f2")
            >>> b
            indexes: [('f2', int64)]
            features: [('f3', int64), ('f1', int64)]
            events:
                f2=1 (4 events):
                    timestamps: [0. 1. 1. 2.]
                    'f3': [1 1 1 1]
                    'f1': [2 1 2 1]
                f2=2 (2 events):
                    timestamps: [1. 1.]
                    'f3': [1 1]
                    'f1': [1 2]
            ...

            >>> # Set both "f1" and "f2" as indices
            >>> b = a.set_index(["f1", "f2"])
            >>> b
            indexes: [('f1', int64), ('f2', int64)]
            features: [('f3', int64)]
            events:
                f1=1 f2=1 (2 events):
                    timestamps: [1. 2.]
                    'f3': [1 1]
                f1=1 f2=2 (1 events):
                    timestamps: [1.]
                    'f3': [1]
                f1=2 f2=1 (2 events):
                    timestamps: [0. 1.]
                    'f3': [1 1]
                f1=2 f2=2 (1 events):
                    timestamps: [1.]
                    'f3': [1]
            ...

            ```

        Args:
            indexes: List of index / feature names (strings) used as
                the new indexes. These names should be either indexes or
                features in the input.

        Returns:
            EventSet with the updated indexes.

        Raises:
            KeyError: If any of the specified `indexes` are not found in the
                input.
        """
        from temporian.core.operators.add_index import set_index

        return set_index(self, indexes=indexes)

    def simple_moving_average(
        self: EventSetOrNode,
        window_length: WindowLength,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the average of values in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each feature independently, returns at
        time t the average value of the feature in the window
        (t - window_length, t].

        `sampling` can't be  specified if a variable `window_length` is
        specified (i.e. if `window_length` is an EventSet).

        If `sampling` is specified or `window_length` is an EventSet, the moving
        window is sampled at each timestamp in them, else it is sampled on the
        input's.

        Missing values (such as NaNs) are ignored.

        If the window does not contain any values (e.g., all the values are
        missing, or the window does not contain any timestamp), outputs missing
        values.

        Example:
            ```python
            >>> a = tp.event_set(
            ...     timestamps=[0, 1, 2, 5, 6, 7],
            ...     features={"value": [np.nan, 1, 5, 10, 15, 20]},
            ... )

            >>> b = a.simple_moving_average(tp.duration.seconds(4))
            >>> b
            indexes: ...
                (6 events):
                    timestamps: [0. 1. 2. 5. 6. 7.]
                    'value': [ nan 1.  3. 7.5  12.5  15. ]
            ...

            ```

        See [`EventSet.moving_count()`][temporian.EventSet.moving_count] for
        examples of moving window operations with external sampling and indices.

        Args:
            window_length: Sliding window's length.
            sampling: Timestamps to sample the sliding window's value at. If not
                provided, timestamps in the input are used.

        Returns:
            EventSet containing the moving average of each feature in the input.
        """
        from temporian.core.operators.window.simple_moving_average import (
            simple_moving_average,
        )

        return simple_moving_average(
            self, window_length=window_length, sampling=sampling
        )

    def since_last(
        self: EventSetOrNode,
        steps: int = 1,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the amount of time since the last previous timestamp in an
        [`EventSet`][temporian.EventSet].

        If a number of `steps` is provided, compute elapsed time after moving
        back that number of previous events.

        Basic example with 1 and 2 steps:
            ```python
            >>> a = tp.event_set(timestamps=[1, 5, 8, 8, 9])

            >>> # Default: time since previous event
            >>> b = a.since_last()
            >>> b
            indexes: ...
                    timestamps: [1. 5. 8. 8. 9.]
                    'since_last': [nan  4.  3.  0.  1.]
            ...

            >>> # Time since 2 previous events
            >>> b = a.since_last(steps=2)
            >>> b
            indexes: ...
                    timestamps: [1. 5. 8. 8. 9.]
                    'since_last': [nan  nan  7.  3.  1.]
            ...

            ```

        If `sampling` is provided, the output will correspond to the time elapsed
        between each timestamp in `sampling` and the latest previous or equal
        timestamp in the input.

        Example with sampling:
            ```python
            >>> a = tp.event_set(timestamps=[1, 4, 5, 7])
            >>> b = tp.event_set(timestamps=[-1, 2, 4, 6, 10])

            >>> # Time elapsed between each sampling event
            >>> # and the latest previous event in a
            >>> c = a.since_last(sampling=b)
            >>> c
            indexes: ...
                    timestamps: [-1. 2. 4. 6. 10.]
                    'since_last': [nan  1.  0.  1. 3.]
            ...

            >>> # 2 steps with sampling
            >>> c = a.since_last(steps=2, sampling=b)
            >>> c
            indexes: ...
                    timestamps: [-1. 2. 4. 6. 10.]
                    'since_last': [nan  nan  3.  2. 5.]
            ...

            ```

        Args:
            steps: Number of previous events to compute elapsed time with.
            sampling: EventSet to use the sampling of.

        Returns:
            Resulting EventSet, with same sampling as `sampling` if provided, or as
                the input if not.
        """
        from temporian.core.operators.since_last import since_last

        return since_last(self, steps=steps, sampling=sampling)

    def tick(
        self: EventSetOrNode, interval: Duration, align: bool = True
    ) -> EventSetOrNode:
        """Generates timestamps at regular intervals in the range of a guide
        [`EventSet`][temporian.EventSet].

        Example with align:
            ```python
            >>> a = tp.event_set(timestamps=[5, 9, 16])
            >>> b = a.tick(interval=tp.duration.seconds(3), align=True)
            >>> b
            indexes: ...
                    timestamps: [ 6. 9. 12. 15.]
            ...

            ```

        Example without align:
            ```python
            >>> a = tp.event_set(timestamps=[5, 9, 16])
            >>> b = a.tick(interval=tp.duration.seconds(3), align=False)
            >>> b
            indexes: ...
                    timestamps: [ 5. 8. 11. 14.]
            ...

            ```

        Args:
            interval: Tick interval.
            align: If false, the first tick is generated at the first timestamp
                (similar to [`EventSet.begin()`][temporian.EventSet.begin]).
                If true (default), ticks are generated on timestamps that are
                multiple of `interval`.

        Returns:
            A feature-less EventSet with regular timestamps.
        """
        from temporian.core.operators.tick import tick

        return tick(self, interval=interval, align=align)

    def tick_calendar(
        self: EventSetOrNode,
        second: Optional[Union[int, Literal["*"]]] = None,
        minute: Optional[Union[int, Literal["*"]]] = None,
        hour: Optional[Union[int, Literal["*"]]] = None,
        mday: Optional[Union[int, Literal["*"]]] = None,
        month: Optional[Union[int, Literal["*"]]] = None,
        wday: Optional[Union[int, Literal["*"]]] = None,
    ) -> EventSetOrNode:
        """Generates events periodically at fixed times or dates e.g. each month.

        Events are generated in the range of the input
        [`EventSet`][temporian.EventSet] independently for each index.

        The usability is inspired in the crontab format, where arguments can
        take a value of `'*'` to tick at all values, or a fixed integer to
        tick only at that precise value.

        Non-specified values (`None`), are set to `'*'` if a finer
        resolution argument is specified, or fixed to the first valid value if
        a lower resolution is specified. For example, setting only
        `tick_calendar(hour='*')`
        is equivalent to:
        `tick_calendar(second=0, minute=0, hour='*', mday='*', month='*')`
        , resulting in one tick at every exact hour of every day/month/year in
        the input guide range.

        The datetime timezone is always assumed to be UTC.

        Examples:
            ```python
            >>> # Every day (at 00:00:00) in the period (exactly one year)
            >>> a = tp.event_set(timestamps=["2021-01-01", "2021-12-31 23:59:59"])
            >>> b = a.tick_calendar(hour=0)
            >>> b
            indexes: ...
            events:
                (365 events):
                    timestamps: [...]
            ...


            >>> # Every day at 2:30am
            >>> b = a.tick_calendar(hour=2, minute=30)
            >>> tp.glue(b.calendar_hour(), b.calendar_minute())
            indexes: ...
            events:
                (365 events):
                    timestamps: [...]
                    'calendar_hour': [2 2 2 ... 2 2 2]
                    'calendar_minute': [30 30 30 ... 30 30 30]
            ...


            >>> # Day 5 of every month (at 00:00)
            >>> b = a.tick_calendar(mday=5)
            >>> b.calendar_day_of_month()
            indexes: ...
            events:
                (12 events):
                    timestamps: [...]
                    'calendar_day_of_month': [5 5 5 ... 5 5 5]
            ...


            >>> # 1st of February of every year
            >>> a = tp.event_set(timestamps=["2020-01-01", "2021-12-31"])
            >>> b = a.tick_calendar(month=2)
            >>> tp.glue(b.calendar_day_of_month(), b.calendar_month())
            indexes: ...
            events:
                (2 events):
                    timestamps: [...]
                    'calendar_day_of_month': [1 1]
                    'calendar_month': [2 2]
            ...

            >>> # Every second in the period  (2 hours -> 7200 seconds)
            >>> a = tp.event_set(timestamps=["2020-01-01 00:00:00",
            ...                              "2020-01-01 01:59:59"])
            >>> b = a.tick_calendar(second='*')
            >>> b
            indexes: ...
            events:
                (7200 events):
                    timestamps: [...]
            ...

            >>> # Every second of the minute 30 of every hour (00:30 and 01:30)
            >>> a = tp.event_set(timestamps=["2020-01-01 00:00",
            ...                              "2020-01-01 02:00"])
            >>> b = a.tick_calendar(second='*', minute=30)
            >>> b
            indexes: ...
            events:
                (120 events):
                    timestamps: [...]
            ...

            >>> # Not allowed: intermediate arguments (minute, hour) not specified
            >>> b = a.tick_calendar(second=1, mday=1)  # ambiguous meaning
            Traceback (most recent call last):
                ...
            ValueError: Can't set argument to None because previous and
            following arguments were specified. Set to '*' or an integer ...

            ```

        Args:
            second: '*' (any second), None (auto) or number in range `[0-59]`
                    to tick at specific second of each minute.
            minute: '*' (any minute), None (auto) or number in range `[0-59]`
                    to tick at specific minute of each hour.
            hour: '*' (any hour), None (auto), or number in range `[0-23]` to
                    tick at specific hour of each day.
            mday: '*' (any day), None (auto) or number in range `[1-31]`
                        to tick at specific day of each month. Note that months
                        without some particular day may not have any tick
                        (e.g: day 31 on February).
            month: '*' (any month), None (auto) or number in range `[1-12]` to
                    tick at one particular month of each year.
            wday: '*' (any day), None (auto) or number in range `[0-6]`
                    (Sun-Sat) to tick at particular day of week. Can only be
                    specified if `day_of_month` is `None`.

        Returns:
            A feature-less EventSet with timestamps at specified interval.
        """
        from temporian.core.operators.tick_calendar import tick_calendar

        return tick_calendar(
            self,
            second=second,
            minute=minute,
            hour=hour,
            mday=mday,
            month=month,
            wday=wday,
        )

    def timestamps(self: EventSetOrNode) -> EventSetOrNode:
        """Converts an [`EventSet`][temporian.EventSet]'s timestamps into a
        `float64` feature.

        Features in the input EventSet are ignored, only the timestamps are used.

        Datetime timestamps are converted to unix timestamps.

        Integer timestamps example:
            ```python
            >>> from datetime import datetime
            >>> a = tp.event_set(timestamps=[1, 2, 3, 5])
            >>> b = a.timestamps()
            >>> b
            indexes: []
            features: [('timestamps', float64)]
            events:
                (4 events):
                    timestamps: [1. 2. 3. 5.]
                    'timestamps': [1. 2. 3. 5.]
            ...

            ```

        Unix timestamps and filter example:
            ```python
            >>> from datetime import datetime
            >>> a = tp.event_set(
            ...    timestamps=[datetime(1970,1,1,0,0,30), datetime(2023,1,1,1,0,0)],
            ... )
            >>> b = a.timestamps()

            >>> # Filter using the timestamps
            >>> max_date = datetime(2020, 1, 1).timestamp()
            >>> c = b.filter(b < max_date)

            >>> # Operate like any other feature
            >>> d = c * 5
            >>> e = tp.glue(c.rename('filtered'), d.rename('multiplied'))
            >>> e
            indexes: []
            features: [('filtered', float64), ('multiplied', float64)]
            events:
                (1 events):
                    timestamps: ['1970-01-01T00:00:30']
                    'filtered': [30.]
                    'multiplied': [150.]
            ...

            ```

        Returns:
            EventSet with a single feature named `timestamps` with each event's
                timestamp.
        """
        from temporian.core.operators.timestamps import timestamps

        return timestamps(self)

    def unique_timestamps(self: EventSetOrNode) -> EventSetOrNode:
        """Removes events with duplicated timestamps from an
        [`EventSet`][temporian.EventSet].

        Returns a feature-less EventSet where each timestamp from the original
        one only appears once. If the input is indexed, the unique operation is
        applied independently for each index.

        Usage example:
            ```python
            >>> a = tp.event_set(timestamps=[5, 9, 9, 16], features={'f': [1,2,3,4]})
            >>> b = a.unique_timestamps()
            >>> b
            indexes: []
            features: []
            events:
                (3 events):
                    timestamps: [ 5. 9. 16.]
            ...

            ```

        Returns:
            EventSet without features with unique timestamps in the input.
        """
        from temporian.core.operators.unique_timestamps import unique_timestamps

        return unique_timestamps(self)

    def until_next(
        self: EventSetOrNode,
        sampling: EventSetOrNode,
        timeout: Duration,
    ) -> EventSetOrNode:
        """Gets the duration until the next sampling event for each input event.

        If no sampling event is observed before `timeout` time-units, returns
        NaN.

        `until_next` is different from `since_last` in that `since_last` returns
        one value for each sampling (sampling events are after input events),
        while `until_next` returns one value for each input value (here again,
        sampling events are after input events).

        The output [`EventSet`][temporian.EventSet] has one event for each event
        in input, but with its timestamp moved forward to the nearest future
        event in `sampling`. If no timestamp in sampling is closer than timeout,
        it is moved by `timeout` into the future instead.

        `until_next` is useful to measure the time it takes for an issue
        (`input`) to be detected by an alert (`sampling`).

        Basic example with 1 and 2 steps:
            ```python
            >>> a = tp.event_set(timestamps=[0, 10, 11, 20, 30])
            >>> b = tp.event_set(timestamps=[1, 12, 21, 22, 42])
            >>> c = a.until_next(sampling=b, timeout=5)
            >>> c
            indexes: []
            features: [('until_next', float64)]
            events:
                (5 events):
                    timestamps: [ 1. 12. 12. 21. 35.]
                    'until_next': [ 1.  2.  1.  1. nan]
            ...

            ```

        Args:
            sampling: EventSet to use the sampling of.
            timeout: Maximum amount of time to wait. If no sampling is observed
                before the timeout expires, the output feature value is NaN.

        Returns:
            Resulting EventSet.
        """
        from temporian.core.operators.until_next import until_next

        return until_next(self, timeout=timeout, sampling=sampling)

    def filter_moving_count(
        self: EventSetOrNode, window_length: Duration
    ) -> EventSetOrNode:
        """Filters out events such that no more than one output event is within
        a tailing time window of `window_length`.

        Filtering is applied in chronological order: An event received at time t
        is filtered out if there is a non-filtered out event in
        (t-window_length, t].

        This operator is different from `(evtset.moving_count(window_length)
        == 0).filter()`. In `filter_moving_count` a filtered event does not
        block following events.

        Usage example:
            ```python
            >>> a = tp.event_set(timestamps=[1, 2, 3])
            >>> b = a.filter_moving_count(window_length=1.5)
            >>> b
            indexes: []
            features: []
            events:
                 (2 events):
                    timestamps: [1. 3.]
            ...

            ```

        Returns:
            EventSet without features with the filtered events.
        """
        from temporian.core.operators.filter_moving_count import (
            filter_moving_count,
        )

        return filter_moving_count(self, window_length=window_length)

    def where(
        self: EventSetOrNode,
        on_true: Union[EventSetOrNode, Any],
        on_false: Union[EventSetOrNode, Any],
    ) -> EventSetOrNode:
        """Choose event-wise feature values from `on_true` or `on_false`
        depending on the boolean value of `self`.

        Given an input [`EventSet`][temporian.EventSet] with a single boolean
        feature, create a new one using the same sampling, and choosing values
        from `on_true` when the input is `True`, otherwise take value from
        `on_false`.

        Both `on_true` and `on_false` can be single values or
        [`EventSets`][temporian.EventSet] with the same sampling as the boolean
        input and one single feature. In any case, both sources must have the
        same data type, or be explicitly casted to the same type beforehand.

        Example with single values:
            ```python
            >>> a = tp.event_set(timestamps=[5, 9, 9],
            ...                  features={'f': [True, True, False]})
            >>> b = a.where(on_true='hello', on_false='goodbye')
            >>> b
            indexes: ...
            events:
                (3 events):
                    timestamps: [5. 9. 9.]
                    'f': [b'hello' b'hello' b'goodbye']
            ...

            ```

        Example with EventSets:
            ```python
            >>> a = tp.event_set(timestamps=[5, 9, 10],
            ...                  features={'condition': [True, True, False],
            ...                            'yes': [1, 2, 3],
            ...                            'no': [-1, -2, -3]})

            >>> b = a['condition'].where(a['yes'], a['no'])
            >>> b
            indexes: ...
            events:
                (3 events):
                    timestamps: [ 5. 9. 10.]
                    'condition': [ 1 2 -3]
            ...

            ```

        Example setting to NaN based on condition:
            ```python
            >>> a = tp.event_set(timestamps=[5, 6, 7, 8, 9],
            ...                  features={'f': [1, 2, -3, -4, 5]})

            >>> # Set values < 0 to nan (cast to float to support nan)
            >>> b = (a['f'] >= 0).where(a['f'].cast(float), np.nan)
            >>> b
            indexes: ...
            events:
                (5 events):
                    timestamps: [5. 6. 7. 8. 9.]
                    'f': [ 1. 2. nan nan 5.]
            ...

            ```

        Args:
            on_true: Source of values from when the condition is True.
            on_false: Source of values from when the condition is False.

        Returns:
            EventSet with a single feature and same sampling as input.
        """
        from temporian.core.operators.where import where

        return where(self, on_true, on_false)
