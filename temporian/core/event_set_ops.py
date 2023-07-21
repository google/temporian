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
from typing import Any, Dict, List, Union, TYPE_CHECKING


if TYPE_CHECKING:
    from temporian.core.typing import EventSetOrNode, TypeOrDType

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
            f"Cannot {op_name} {self._clsname} and {type(other)} objects. Only"
            f" {self._clsname} or values of type ({allowed_types}) are"
            " supported."
        )

    def __ne__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import not_equal

            return not_equal(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR + (bool, str)):
            from temporian.core.operators.scalar import not_equal_scalar

            return not_equal_scalar(input=self, value=other)

        self._raise_error("ne", other, "int,float,bool,str")
        assert False

    def __add__(self, other: Any):
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

    def __sub__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import subtract

            return subtract(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                subtract_scalar,
            )

            return subtract_scalar(minuend=self, subtrahend=other)

        self._raise_error("subtract", other, "int,float")
        assert False

    def __rsub__(self, other: Any):
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                subtract_scalar,
            )

            return subtract_scalar(minuend=other, subtrahend=self)

        self._raise_error("subtract", other, "int,float")
        assert False

    def __mul__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import multiply

            return multiply(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                multiply_scalar,
            )

            return multiply_scalar(input=self, value=other)

        self._raise_error("multiply", other, "int,float")
        assert False

    def __rmul__(self, other: Any):
        return self.__mul__(other)

    def __neg__(self):
        from temporian.core.operators.scalar import multiply_scalar

        return multiply_scalar(input=self, value=-1)

    def __invert__(self):
        from temporian.core.operators.unary import invert

        return invert(input=self)

    def __abs__(self):
        from temporian.core.operators.unary import abs

        return abs(input=self)

    def __truediv__(self, other: Any):
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

    def __floordiv__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import floordiv

            return floordiv(numerator=self, denominator=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=self, denominator=other)

        self._raise_error("floor_divide", other, "(int,float)")
        assert False

    def __rfloordiv__(self, other: Any):
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=other, denominator=self)

        self._raise_error("floor_divide", other, "(int,float)")
        assert False

    def __pow__(self, other: Any):
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

    def __mod__(self, other: Any):
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

    def __gt__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import greater

            return greater(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                greater_scalar,
            )

            return greater_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def __ge__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import greater_equal

            return greater_equal(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                greater_equal_scalar,
            )

            return greater_equal_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def __lt__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import less

            return less(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                less_scalar,
            )

            return less_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def __le__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import less_equal

            return less_equal(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                less_equal_scalar,
            )

            return less_equal_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def _raise_bool_error(self, boolean_op: str, other: Any) -> None:
        raise ValueError(
            f"Cannot compute '{self._clsname} {boolean_op} {type(other)}'. "
            f"Only {self._clsname}s with boolean features are supported."
        )

    def __and__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import logical_and

            return logical_and(input_1=self, input_2=other)

        self._raise_bool_error("&", other)
        assert False

    def __or__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import logical_or

            return logical_or(input_1=self, input_2=other)

        self._raise_bool_error("|", other)
        assert False

    def __xor__(self, other: Any):
        if isinstance(other, self.__class__):
            from temporian.core.operators.binary import logical_xor

            return logical_xor(input_1=self, input_2=other)

        self._raise_bool_error("^", other)
        assert False

    #############
    # OPERATORS #
    #############

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
                indexes. These feature names should already exist in `input`.

        Returns:
            EventSet with the extended index.

        Raises:
            KeyError: If any of the specified `indexes` are not found in `input`.
        """
        from temporian.core.operators.add_index import add_index

        return add_index(self, indexes)

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

    def cast(
        self: EventSetOrNode,
        target: Union[
            TypeOrDType,
            Dict[str, TypeOrDType],
            Dict[TypeOrDType, TypeOrDType],
        ],
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

        return cast(self, target, check_overflow)

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
                the new indexes. These names should be either indexes or features in
                `input`.

        Returns:
            EventSet with the updated indexes.

        Raises:
            KeyError: If any of the specified `indexes` are not found in `input`.
        """
        from temporian.core.operators.add_index import set_index

        return set_index(self, indexes)