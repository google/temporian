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

"""Utility functions to compute Durations from other units of time.

Timestamps and durations are expressed with a double (noted float) in python."""
from typing import Union

from temporian.core.data.duration_utils import NormalizedDuration, Duration


def milliseconds(value: Union[int, float]) -> Duration:
    """Converts input value from milliseconds to a `Duration` in seconds.

    Args:
        value: Number of milliseconds.

    Returns:
        Equivalent number of seconds.
    """
    return NormalizedDuration(value / 1000)


def seconds(value: Union[int, float]) -> Duration:
    """Converts input value from seconds to a `Duration` in seconds.

    Since the `Duration` object is equivalent to a `float` value in seconds,
    this method does nothing else than casting the input to `float`. It may be
    used in order to make the code more explicit.

    Args:
        value: Number of seconds.

    Returns:
        Same number of seconds.
    """
    return NormalizedDuration(value)


def minutes(value: Union[int, float]) -> Duration:
    """Converts input value from minutes to a `Duration` in seconds.

    Args:
        value: Number of minutes.

    Returns:
        Equivalent number of seconds.
    """
    return NormalizedDuration(value * 60)


def hours(value: Union[int, float]) -> Duration:
    """Converts input value from hours to a `Duration` in seconds.

    Args:
        value: Number of hours.

    Returns:
        Equivalent number of seconds.
    """
    return NormalizedDuration(value * 60 * 60)


def days(value: Union[int, float]) -> Duration:
    """Converts input value from number of days to a `Duration` in seconds.

    Args:
        value: number of days.

    Returns:
        Equivalent number of seconds.
    """
    return NormalizedDuration(value * 60 * 60 * 24)


def weeks(value: Union[int, float]) -> Duration:
    """Converts input value from number of weeks to a `Duration` in seconds.

    Args:
        value: Number of weeks.

    Returns:
        Equivalent number of seconds.
    """
    return NormalizedDuration(value * 60 * 60 * 24 * 7)
