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

from typing import Union

# NOTE: this module is shown in the docs and so will any symbol in this file
# with a docstring and a non-private name.


Duration = Union[float, int]
"""A duration in seconds, stored as a float64.

In Temporian, timestamps and durations are float64 values, and it is up to the
user to choose the semantic of this value.

However, some functions are datetime-related (such as the functions defined in
this module, calendar operators, plotting functions, and more) and assume that
durations are expressed in seconds (see
[Time units](https://temporian.readthedocs.io/en/latest/user_guide/#time-units)),
so it is recommended to use seconds as timestamps where possible.
"""


def milliseconds(value: Union[int, float]) -> Duration:
    """Converts input value from milliseconds to a `Duration` in seconds.

    Example:
        ```python
        >>> duration = tp.duration.milliseconds(250)
        >>> duration
        0.25

        >>> # Usage in a window operation
        >>> a = tp.event_set(
        ...     timestamps=[0.5, 1.0, 1.2],
        ...     features={"f1": [1, 5, -5]}
        ... )
        >>> tp.moving_sum(a, window_length=duration)
        indexes: ...
                timestamps: [0.5 1.  1.2]
                'f1': [1 5 0]
        ...

        ```

    Args:
        value: Number of milliseconds.

    Returns:
        Equivalent number of seconds.
    """
    return float(value / 1000)


def seconds(value: Union[int, float]) -> Duration:
    """Converts input value from seconds to a `Duration` in seconds.

    Since the `Duration` object is equivalent to a `float` value in seconds,
    this method does nothing else than casting the input to `float`. It may be
    used in order to make the code more explicit.

    Explicit time units:
        ```python
        >>> duration = tp.duration.seconds(3)
        >>> duration
        3.0

        >>> # Usage in a window operation
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 6],
        ...     features={"f1": [1, 5, -5]},
        ... )
        >>> tp.moving_sum(a, window_length=duration)
        indexes: ...
                timestamps: [1. 2. 6.]
                'f1': [ 1 6 -5]
        ...

        ```

    Args:
        value: Number of seconds.

    Returns:
        Same number of seconds.
    """
    return float(value)


def minutes(value: Union[int, float]) -> Duration:
    """Converts input value from minutes to a `Duration` in seconds.

    Example:
        ```python
        >>> timestamps = [tp.duration.minutes(i) for i in [5, 10, 30]]
        >>> timestamps
        [300.0, 600.0, 1800.0]

        >>> # Usage in a window operation
        >>> a = tp.event_set(timestamps=timestamps, features={"f1": [1, 5, -5]})
        >>> tp.moving_sum(a, window_length=tp.duration.minutes(6))
        indexes: ...
                timestamps: [ 300. 600. 1800.]
                'f1': [ 1 6 -5]
        ...

        ```

    Args:
        value: Number of minutes.

    Returns:
        Equivalent number of seconds.
    """
    return float(value * 60)


def hours(value: Union[int, float]) -> Duration:
    """Converts input value from hours to a `Duration` in seconds.

    Example:
        ```python
        >>> timestamps = [tp.duration.hours(i) for i in [1, 2, 10]]
        >>> timestamps
        [3600.0, 7200.0, 36000.0]

        >>> # Usage in a window operation
        >>> a = tp.event_set(timestamps=timestamps, features={"f1": [1, 5, -5]})
        >>> tp.moving_sum(a, window_length=tp.duration.hours(2))
        indexes: ...
                timestamps: [ 3600. 7200. 36000.]
                'f1': [ 1 6 -5]
        ...

        ```

    Args:
        value: Number of hours.

    Returns:
        Equivalent number of seconds.
    """
    return float(value * 60 * 60)


def days(value: Union[int, float]) -> Duration:
    """Converts input value from number of days to a `Duration` in seconds.

    Example:
        ```python
        >>> a = tp.event_set(
        ...    # Dates are converted to unix timestamps
        ...    timestamps=["2020-01-01", "2020-01-02", "2020-01-31"],
        ...    features={"f1": [1, 5, -5]}
        ... )

        >>> tp.moving_sum(a, window_length=tp.duration.days(2))
        indexes: ...
                timestamps: [1.5778e+09 1.5779e+09 1.5804e+09]
                'f1': [ 1 6 -5]
        ...

        ```

    Args:
        value: number of days.

    Returns:
        Equivalent number of seconds.
    """
    return float(value * 60 * 60 * 24)


def weeks(value: Union[int, float]) -> Duration:
    """Converts input value from number of weeks to a `Duration` in seconds.

        ```python
        >>> a = tp.event_set(
        ...    # Dates are converted to unix timestamps
        ...    timestamps=["2020-01-01", "2020-01-07", "2020-01-31"],
        ...    features={"f1": [1, 5, -5]}
        ... )

        >>> tp.moving_sum(a, window_length=tp.duration.weeks(2))
        indexes: ...
                timestamps: [1.5778e+09 1.5784e+09 1.5804e+09]
                'f1': [ 1 6 -5]
        ...

        ```

    Args:
        value: Number of weeks.

    Returns:
        Equivalent number of seconds.
    """
    return float(value * 60 * 60 * 24 * 7)
