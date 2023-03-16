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
from typing import Union

import datetime
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
