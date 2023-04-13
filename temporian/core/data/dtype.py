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


from typing import Union, Any
import math

FLOAT64 = "FLOAT64"
FLOAT32 = "FLOAT32"
INT64 = "INT64"
INT32 = "INT32"
STRING = "STRING"
BOOLEAN = "BOOLEAN"

DType = Union[FLOAT64, FLOAT32, INT64, INT32, STRING, BOOLEAN]

ALL_TYPES = [FLOAT64, FLOAT32, INT64, INT32, STRING, BOOLEAN]


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


def python_type_to_temporian_dtype(python_type: Any) -> DType:
    """
    Convert a Python type to its corresponding Temporian dtype.

    Args:
        python_type: The Python type to be converted.

    Returns:
        The corresponding Temporian dtype.

    Raises:
        ValueError: If the input Python type is not supported.
    """
    if issubclass(python_type, float):
        return FLOAT64  # Assuming default float type is FLOAT64
    if issubclass(python_type, int):
        return INT64  # Assuming default int type is INT64
    if issubclass(python_type, str):
        return STRING
    if issubclass(python_type, bool):
        return BOOLEAN

    raise ValueError(f"Unsupported Python type: {python_type}")


def same_kind(dtype1: DType, dtype2: DType) -> bool:
    """
    Check if two dtypes are of the same kind.

    Args:
        dtype1: The first dtype.
        dtype2: The second dtype.

    Returns:
        True if the two dtypes are of the same kind, False otherwise.
    """
    floats = [FLOAT64, FLOAT32]
    ints = [INT64, INT32]

    if dtype1 in floats and dtype2 in floats:
        return True

    if dtype1 in ints and dtype2 in ints:
        return True

    if dtype1 == STRING and dtype2 == STRING:
        return True

    if dtype1 == BOOLEAN and dtype2 == BOOLEAN:
        return True

    return False
