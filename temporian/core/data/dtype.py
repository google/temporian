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

"""Data types declaration."""

from typing import Union, Any
import math

FLOAT64 = "FLOAT64"
FLOAT32 = "FLOAT32"
INT64 = "INT64"
INT32 = "INT32"
STRING = "STRING"

DType = Union[FLOAT64, FLOAT32, INT64, INT32, STRING]

ALL_TYPES = [FLOAT64, FLOAT32, INT64, INT32, STRING]


def MissingValue(dtype: DType) -> Any:
    """Value used as a replacement of missing values."""

    if dtype in [FLOAT64, FLOAT32]:
        return math.nan
    elif dtype in [INT64, INT32]:
        return 0
    elif dtype == STRING:
        return ""
    else:
        raise ValueError(f"Non implemented type {dtype}")
