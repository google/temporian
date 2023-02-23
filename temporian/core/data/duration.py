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
from typing import Union

import numpy as np

# Unit for durations
Duration = float


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


def convert_date_to_duration(
    date: Union[np.datetime64, datetime.datetime]
) -> Duration:
    """Convert date to duration epoch UTC

    Args:
        date (Union[np.datetime64, datetime.datetime]):
            Date to convert

    Returns:
        int: Duration epoch UTC

    Raises:
        TypeError: Unsupported type. Supported types are:
        - np.datetime64
        - datetime.datetime
    """
    # if it is already a duration, return it
    if isinstance(date, float):
        return date
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


def duration_abbreviation(duration: Duration, cutoff: str = None) -> str:
    """Returns the abbreviation for a duration.

    Args:
        duration: Duration in seconds.
        cutoff: Cutoff for the abbreviation. For example, if cutoff is "day", the
        smallest unit will be days. Possible options are "year", "month", "week",
        "day", "hour" and "minute". If None, the smallest unit will be seconds.

    Returns:
        Abbreviation for the duration.
    """

    duration_str = ""

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
        duration = duration % seconds(1)

    return duration_str
