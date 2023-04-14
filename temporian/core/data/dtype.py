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
    BOOLEAN = "BOOLEAN"

    def __str__(self) -> str:
        return self.value

    @property
    def is_float(self) -> bool:
        return self in (DType.FLOAT64, DType.FLOAT32)

    @property
    def is_integer(self) -> bool:
        return self in (DType.INT64, DType.INT32)

    def missing_value(self) -> Union[float, int, str]:
        """
        Returns missing value for specific dtype.

        Returns:
            The default missing value for the given data type.
        """

        if self.is_float:
            return math.nan

        elif self.is_integer:
            return 0

        elif self == DType.STRING:
            return ""

        else:
            raise ValueError(f"Non-implemented type {self}")
