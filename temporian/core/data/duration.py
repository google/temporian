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
