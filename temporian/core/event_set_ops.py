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
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from temporian.core.data.duration import Duration


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

    def calendar_day_of_month(self: EventSetOrNode) -> EventSetOrNode:
        """Obtains the day of month the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 1 and 31.

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

        Returns:
            EventSet with a single feature with the day of the month each timestamp
            in `sampling` belongs to.
        """
        from temporian.core.operators.calendar.day_of_month import (
            calendar_day_of_month,
        )

        return calendar_day_of_month(self)

    def calendar_day_of_week(self: EventSetOrNode) -> EventSetOrNode:
        """Obtains the day of the week the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers from 0 (Monday) to 6 (Sunday).

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

        Returns:
            EventSet with a single feature with the day of the week each timestamp
            in `sampling` belongs to.
        """
        from temporian.core.operators.calendar.day_of_week import (
            calendar_day_of_week,
        )

        return calendar_day_of_week(self)

    def calendar_hour(self: EventSetOrNode) -> EventSetOrNode:
        """Obtains the hour the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 0 and 23.

        Usage example:
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

        Returns:
            EventSet with a single feature with the hour each timestamp in `sampling`
            belongs to.
        """
        from temporian.core.operators.calendar.hour import calendar_hour

        return calendar_hour(self)

    def calendar_iso_week(self: EventSetOrNode) -> EventSetOrNode:
        """Obtains the ISO week the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 1 and 53.

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

        Returns:
            EventSet with a single feature with the ISO week each timestamp in
            `sampling` belongs to.
        """
        from temporian.core.operators.calendar.iso_week import calendar_iso_week

        return calendar_iso_week(self)

    def calendar_day_of_year(self: EventSetOrNode) -> EventSetOrNode:
        """Obtains the day of year the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 1 and 366.

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

        Returns:
            EventSet with a single feature with the day of the year each timestamp
            in `sampling` belongs to.
        """
        from temporian.core.operators.calendar.day_of_year import (
            calendar_day_of_year,
        )

        return calendar_day_of_year(self)

    def calendar_minute(self: EventSetOrNode) -> EventSetOrNode:
        """Obtain the minute the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between
        0 and 59.

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

        Returns:
            EventSet with a single feature with the minute each timestamp in
            `sampling` belongs to.
        """
        from temporian.core.operators.calendar.minute import calendar_minute

        return calendar_minute(self)

    def calendar_month(self: EventSetOrNode) -> EventSetOrNode:
        """Obtains the month the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 1 and 12.

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

        Returns:
            EventSet with a single feature with the month each timestamp in
            `sampling` belongs to.
        """
        from temporian.core.operators.calendar.month import calendar_month

        return calendar_month(self)

    def calendar_second(self: EventSetOrNode) -> EventSetOrNode:
        """Obtains the second the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

        Output feature contains numbers between 0 and 59.

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

        Returns:
            EventSet with a single feature with the second each timestamp in
            `sampling` belongs to.
        """
        from temporian.core.operators.calendar.second import calendar_second

        return calendar_second(self)

    def calendar_year(self: EventSetOrNode) -> EventSetOrNode:
        """Obtains the year the timestamps in an
        [`EventSet`][temporian.EventSet]'s sampling are in.

        Features in the input are ignored, only the timestamps are used and
        they must be unix timestamps (`is_unix_timestamp=True`).

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

        Returns:
            EventSet with a single feature with the year each timestamp in
            `sampling` belongs to.
        """
        from temporian.core.operators.calendar.year import calendar_year

        return calendar_year(self)

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

        return cast(self, target=target, check_overflow=check_overflow)

    def cumsum(
        self: EventSetOrNode,
    ) -> EventSetOrNode:
        """Computes the cumulative sum of values over each feature in an
        [`EventSet`][temporian.EventSet].

        Foreach timestamp, calculate the sum of the feature from the beginning.
        Shorthand for `moving_sum(event, window_length=np.inf)`.

        Missing values are ignored.

        While the feature does not have any values (e.g., missing initial values),
        outputs missing values.

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

        Returns:
            Cumulative sum of each feature in `node`.
        """
        from temporian.core.operators.window.moving_sum import cumsum

        return cumsum(self)

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

        Returns
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

    def moving_count(
        self: EventSetOrNode,
        window_length: Duration,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Gets the number of events in a sliding window.

        Create a tp.int32 feature containing the number of events in the time
        window (t - window_length, t].

        If the `sampling` argument is not provided, outputs a timestamp for
        each timestamp in `input`. If the `sampling` argument is provided,
        outputs a timestamp for each timestamp in `sampling`.

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
        window_length: Duration,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the maximum in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each index and feature independently,
        returns at time t the max of non-nan values for the feature in the window
        (t - window_length, t].

        If `sampling` is provided samples the moving window's value at each
        timestamp in `sampling`, else samples it at each timestamp in the input.

        If the window does not contain any values (e.g., all the values are missing,
        or the window does not contain any sampling), outputs missing values.

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
        window_length: Duration,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the minimum of values in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each index and feature independently,
        returns at time t the minimum of non-nan values for the feature in the window
        (t - window_length, t].

        If `sampling` is provided samples the moving window's value at each
        timestamp in `sampling`, else samples it at each timestamp in the input.

        If the window does not contain any values (e.g., all the values are missing,
        or the window does not contain any sampling), outputs missing values.

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

        See [`EventSet.moving_count()`][temporian.EventSet.moving_count] for examples
        of moving window operations with external sampling and indices.

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
        window_length: Duration,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the standard deviation of values in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each feature independently, returns at time
        t the standard deviation for the feature in the window
        (t - window_length, t].

        If `sampling` is provided samples the moving window's value at each
        timestamp in `sampling`, else samples it at each timestamp in the input.

        Missing values (such as NaNs) are ignored.

        If the window does not contain any values (e.g., all the values are missing,
        or the window does not contain any sampling), outputs missing values.

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

        See [`EventSet.moving_count()`][temporian.EventSet.moving_count] for examples of moving window
        operations with external sampling and indices.

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
        window_length: Duration,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the sum of values in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each feature independently, returns at time
        t the sum of the feature in the window (t - window_length, t].

        If `sampling` is provided samples the moving window's value at each
        timestamp in `sampling`, else samples it at each timestamp in the input.

        Missing values (such as NaNs) are ignored.

        If the window does not contain any values (e.g., all the values are missing,
        or the window does not contain any sampling), outputs missing values.

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

        See [`EventSet.moving_count()`][temporian.EventSet.moving_count] for examples of moving window
        operations with external sampling and indices.

        Args:
            window_length: Sliding window's length.
            sampling: Timestamps to sample the sliding window's value at. If not
                provided, timestamps in the input are used.

        Returns:
            EventSet containing the moving sum of each feature in the input.
        """
        from temporian.core.operators.window.moving_sum import moving_sum

        return moving_sum(self, window_length=window_length, sampling=sampling)

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
                the input.

        Returns:
            EventSet with the updated indexes.

        Raises:
            KeyError: If any of the specified `indexes` are not found in the input.
        """
        from temporian.core.operators.add_index import set_index

        return set_index(self, indexes=indexes)

    def simple_moving_average(
        self: EventSetOrNode,
        window_length: Duration,
        sampling: Optional[EventSetOrNode] = None,
    ) -> EventSetOrNode:
        """Computes the average of values in a sliding window over an
        [`EventSet`][temporian.EventSet].

        For each t in sampling, and for each feature independently, returns at time
        t the average value of the feature in the window (t - window_length, t].

        If `sampling` is provided samples the moving window's value at each
        timestamp in `sampling`, else samples it at each timestamp in the input.

        Missing values (such as NaNs) are ignored.

        If the window does not contain any values (e.g., all the values are missing,
        or the window does not contain any sampling), outputs missing values.

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

        See [`EventSet.moving_count()`][temporian.EventSet.moving_count] for examples of moving window
        operations with external sampling and indices.

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
        second: Union[int, str, None] = None,
        minute: Union[int, str, None] = None,
        hour: Union[int, str, None] = None,
        day_of_month: Union[int, str, None] = None,
        month: Union[int, str, None] = None,
        day_of_week: Union[int, str, None] = None,
    ) -> EventSetOrNode:
        """Generates timestamps at specified datetimes, in the range of a guide
        [`EventSet`][temporian.EventSet].

        The usability is inspired in the crontab format, where arguments can
        take a value of `'*'` to tick at all values, or a fixed integer to
        tick only at that precise value.

        Non-specified values (`None`), are set to `*` if a finer
        resolution argument is specified, or fixed to the first valid value if
        a lower resolution is specified. For example, setting only
        `tick_calendar(hour='*')`
        is equivalent to:
        `tick_calendar(second=0, minute=0, hour='*', day_of_month='*', month='*')`
        , resulting in one tick at every exact hour of every day/month/year in
        the input guide range.

        Example:
            ```python
            >>> a = tp.event_set(timestamps=["2020-01-01", "2021-01-01"])
            >>> # Every day in the period
            >>> b = a.tick_calendar()
            >>> b

            >>> # Every day at 2:30am
            >>> b = a.tick_calendar(hour=2, minute=30)
            >>> b

            >>> # Day 5 of every month
            >>> b = a.tick_calendar(day_of_month=5)
            >>> b

            >>> a = tp.event_set(timestamps=["2020-01-01", "2023-01-01"])
            >>> # 1st of February of every year
            >>> b = a.tick_calendar(month=2)
            >>> b

            ```

        Args:
            second: '*' (any second), None (auto) or number in range `[0-59]`
                    to tick at specific second of each minute.
            minute: '*' (any minute), None (auto) or number in range `[0-59]`
                    to tick at specific minute of each hour.
            hour: '*' (any hour), None (auto), or number in range `[0-23]` to
                    tick at specific hour of each day.
            day_of_month: '*' (any day), None (auto) or number in range `[1-31]`
                        to tick at specific day of each month. Note that months
                        without some particular day may not have any tick
                        (e.g: day 31 on February).
            month: '*' (any month), None (auto) or number in range `[1-12]` to
                    tick at one particular month of each year.
            day_of_week: '*' (any day), None (auto) or number in range `[0-6]`
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
            day_of_month=day_of_month,
            month=month,
            day_of_week=day_of_week,
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
                    timestamps: [30.]
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
