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
"""
from typing import Union

import datetime
import time
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


supported_date_types = (np.datetime64, time.struct_time, datetime.datetime)


def convert_date_to_duration(
    date: Union[np.datetime64, time.struct_time, datetime.datetime]
) -> Duration:
    """Convert date to duration epoch UTC

    Args:
        date (Union[np.datetime64, time.struct_time, datetime.datetime]):
            Date to convert

    Returns:
        int: Duration epoch UTC

    Raises:
        TypeError: Unsupported type. Supported types are:
        - np.datetime64
        - time.struct_time
        - datetime.datetime
    """
    if isinstance(date, np.datetime64):
        return convert_numpy_datetime64_to_duration(date)
    if isinstance(date, time.struct_time):
        return convert_struct_time_to_duration(date)
    if isinstance(date, datetime.datetime):
        return convert_datetime_to_duration(date)

    raise TypeError(f"Unsupported type: {type(date)}")


def convert_numpy_datetime64_to_duration(date: np.datetime64) -> Duration:
    """Convert numpy datetime64 to duration epoch UTC"""
    return date.astype("datetime64[s]").astype("float64")


def convert_struct_time_to_duration(date: time.struct_time) -> Duration:
    """Convert struct_time to duration epoch UTC"""
    return float(time.mktime(date) - time.timezone)


def convert_datetime_to_duration(date: datetime.datetime) -> Duration:
    """Convert datetime to duration epoch UTC"""
    date = date.replace(tzinfo=datetime.timezone.utc)
    return float(date.timestamp())


def is_a_date(value: any) -> bool:
    """Check if the value is a supported date type.

    Args:
        value (any): Value to check

    Returns:
        bool: True if the value is a date, False otherwise
    """
    for date_type in supported_date_types:
        if value is date_type:
            return True
    return False
