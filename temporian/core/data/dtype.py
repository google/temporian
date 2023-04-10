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

"""Type of features."""


import math
from enum import Enum
from typing import Union


class DType(Enum):
    FLOAT64 = "FLOAT64"
    FLOAT32 = "FLOAT32"
    INT64 = "INT64"
    INT32 = "INT32"
    STRING = "STRING"

    def __str__(self) -> str:
        return self.value


def MissingValue(dtype: DType) -> Union[float, int, str]:
    """Value used as a replacement of missing values."""

    if dtype in [DType.FLOAT64, DType.FLOAT32]:
        return math.nan
    elif dtype in [DType.INT64, DType.INT32]:
        return 0
    elif dtype == DType.STRING:
        return ""
    else:
        raise ValueError(f"Non implemented type {dtype}")
