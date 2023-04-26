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

"""Utility functions to handle durations.

Timestamples and durations are expressed with a double (noted float) in python.
By convension, all calendar functions represent dates as Unix epoch in UTC.
This datatype is equivalent to a double in C.
"""
import datetime
from enum import Enum
from typing import Union

import numpy as np

# Unit for durations
Duration = float

Timestamp = Union[np.datetime64, datetime.datetime, int, float]


def milliseconds(value: float) -> Duration:
    return value / 1000


def seconds(value: float) -> Duration:
    return value


def minutes(value: float) -> Duration:
    return value * 60


def hours(value: float) -> Duration:
    return value * 60 * 60


def days(value: float) -> Duration:
    return value * 60 * 60 * 24


def weeks(value: float) -> Duration:
    return value * 60 * 60 * 24 * 7


def convert_date_to_duration(date: Timestamp) -> Duration:
    """Convert date value to float.

    If a float or int, it is returned as float.
    If a date, it is converted to a Unix timestamp (number of seconds from Unix
    epoch).

    Args:
        date: date to convert.

    Returns:
        int: unix timestamp (seconds elapsed from unix epoch).

    Raises:
        TypeError: unsupported type. Supported types are:
            - np.datetime64
            - datetime.datetime
    """
    # if it is already a number, return it as float
    if isinstance(date, float):
        return date
    if isinstance(date, int):
        return float(date)

    # if it is a date, convert it to unix timestamp
    if isinstance(date, np.datetime64):
        return convert_numpy_datetime64_to_duration(date)
    if isinstance(date, datetime.datetime):
        return convert_datetime_to_duration(date)

    raise TypeError(f"Unsupported type: {type(date)}")


def convert_numpy_datetime64_to_duration(date: np.datetime64) -> Duration:
    """Convert numpy datetime64 to duration epoch UTC"""
    return date.astype("datetime64[s]").astype("float64")


def convert_datetime_to_duration(date: datetime.datetime) -> Duration:
    """Convert datetime to duration epoch UTC"""
    return date.replace(tzinfo=datetime.timezone.utc).timestamp()


class TimeUnit(str, Enum):
    """Time unit for a duration."""

    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"

    @staticmethod
    def is_valid(value: any) -> bool:
        return isinstance(value, TimeUnit) or (
            isinstance(value, str)
            and value in [item.value for item in TimeUnit]
        )


def duration_abbreviation(
    duration: Duration, cutoff: Union[str, TimeUnit] = "milliseconds"
) -> str:
    """Returns the abbreviation for a duration.

    Args:
        duration: Duration in seconds.
        cutoff: Cutoff for the abbreviation. For example, if cutoff is "day", the
            smallest unit will be days. Possible options are "week",
            "day", "hour" and "minute", "seconds" and "milliseconds". Default is
            "milliseconds".

    Returns:
        Abbreviation for the duration.
    """
    # check cuttoff is a TimeUnit or if its a string that is a valid TimeUnit
    if not TimeUnit.is_valid(cutoff):
        raise ValueError(
            f"Invalid cutoff: {cutoff}. Possible options are: {list(TimeUnit)}"
        )

    duration_str = ""

    if duration < 0:
        duration = -duration

    if duration >= weeks(1):
        duration_str += f"{int(duration / weeks(1))}w"
        if cutoff == "week":
            return duration_str
        duration = duration % weeks(1)

    if duration >= days(1):
        duration_str += f"{int(duration / days(1))}d"
        if cutoff == "day":
            return duration_str
        duration = duration % days(1)

    if duration >= hours(1):
        duration_str += f"{int(duration / hours(1))}h"
        if cutoff == "hour":
            return duration_str
        duration = duration % hours(1)

    if duration >= minutes(1):
        duration_str += f"{int(duration / minutes(1))}min"
        if cutoff == "minute":
            return duration_str
        duration = duration % minutes(1)

    if duration >= seconds(1):
        duration_str += f"{int(duration / seconds(1))}s"
        if cutoff == "seconds":
            return duration_str
        duration = duration % seconds(1)

    if duration >= milliseconds(1):
        duration_str += f"{int(duration / milliseconds(1))}ms"
        return duration_str

    return duration_str
