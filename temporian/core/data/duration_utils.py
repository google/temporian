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

Timestamps and durations are expressed with a double (noted float) in python.
By convention, all calendar functions represent dates as Unix epoch in UTC.
This datatype is equivalent to a double in C.
"""
import datetime
from enum import Enum
from typing import Union, Iterable, List

import numpy as np

from temporian.core.data.duration import (
    Duration,
    weeks,
    days,
    hours,
    minutes,
    seconds,
    milliseconds,
)


# Unit for durations
#
# NormalizedDuration is used by internal code that handle duration.
# Duration (defined in ./duration.py) is a duration provided by the user though the API.
NormalizedDuration = float

Timestamp = Union[np.datetime64, datetime.datetime, int, float]
NormalizedTimestamp = float


def normalize_duration(x: Duration) -> NormalizedDuration:
    if isinstance(x, (int, float)) and x > 0:
        return NormalizedDuration(x)

    raise ValueError(
        "A duration should be a strictly positive number of type float,"
        f" int or tp.Duration. Got {x!r} of type {type(x)}."
    )


def normalize_timestamp(x: Timestamp) -> NormalizedTimestamp:
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[ns]").astype(np.float64) / 1e9

    if isinstance(x, datetime.datetime):
        return float(x.timestamp())

    if isinstance(x, int):
        return float(x)

    if isinstance(x, float):
        return x

    raise ValueError(f"Invalid timestamp {x!r} of type {type(x)}.")


def convert_timestamp_to_datetime(timestamp: Timestamp) -> datetime.datetime:
    """Converts a unix timestamp in seconds to datetime (UTC).

    Args:
        timestamp: Single timestamp in seconds.

    Returns:
        Single UTC datetime.
    """
    norm_timestamp = normalize_timestamp(timestamp)
    return datetime.datetime.fromtimestamp(
        norm_timestamp, tz=datetime.timezone.utc
    )


def convert_timestamps_to_datetimes(
    ts: Iterable[Timestamp],
) -> List[datetime.datetime]:
    """Converts unix timestamps in seconds to a list of datetimes (UTC).

    Args:
        ts: Iterable of timestamps, in seconds.

    Returns:
        List of UTC datetimes.
    """
    return [convert_timestamp_to_datetime(t) for t in ts]


def convert_date_to_duration(date: Timestamp) -> NormalizedDuration:
    """Converts date value to a number representing the Unix timestamp.

    If a float or int, it is returned as float.
    If a date, it is converted to a Unix timestamp (number of seconds from Unix
    epoch).

    Args:
        date: Date to convert.

    Returns:
        Unix timestamp (seconds elapsed from unix epoch).

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
    if isinstance(date, datetime.date):
        return convert_datetime_date_to_duration(date)

    raise TypeError(f"Unsupported type: {type(date)}")


def convert_numpy_datetime64_to_duration(
    date: np.datetime64,
) -> NormalizedDuration:
    """Convert numpy datetime64 to duration epoch UTC."""
    return float(date.astype("datetime64[s]").astype("float64"))


def convert_datetime_to_duration(date: datetime.datetime) -> NormalizedDuration:
    """Convert datetime to duration epoch UTC."""
    return float(date.replace(tzinfo=datetime.timezone.utc).timestamp())


def convert_datetime_date_to_duration(
    date: datetime.date,
) -> NormalizedDuration:
    """Convert date to duration epoch UTC."""
    return convert_datetime_to_duration(
        datetime.datetime.combine(date, datetime.time(0, 0))
    )


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
        cutoff: Cutoff for the abbreviation. For example, if cutoff is "day",
            the smallest unit will be days. Possible options are "week",
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
