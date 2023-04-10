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

DType = Union[FLOAT64, FLOAT32, INT64, INT32, STRING]


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


def get_resulting_dtype(dtype1: DType, dtype2: DType) -> DType:
    """
    Get the resulting dtype when combining two arrays with the specified dtypes.

    This function takes two dtypes as input and returns the expected
    output dtype based on a simplified dtype hierarchy.

    Args:
        dtype1: The dtype of the first array.
        dtype2: The dtype of the second array.

    Returns:
        DType: The resulting dtype

    Raises:
        ValueError: If either of the input dtypes is not supported.
    """
    dtype_hierarchy = {
        INT32: 1,
        FLOAT32: 2,
        INT64: 3,
        FLOAT64: 4,
    }

    if dtype1 not in dtype_hierarchy or dtype2 not in dtype_hierarchy:
        supported_dtypes = ", ".join(dtype_hierarchy.keys())
        raise ValueError(
            f"Invalid dtype(s): [{dtype1}, {dtype2}]. Supported dtypes:"
            f" {supported_dtypes}"
        )

    # Find the highest hierarchy value between the two input dtypes
    max_hierarchy = max(dtype_hierarchy[dtype1], dtype_hierarchy[dtype2])

    # handle special case with int64 and float32
    if (dtype1 == INT64 and dtype2 == FLOAT32) or (
        dtype1 == FLOAT32 and dtype2 == INT64
    ):
        return FLOAT64

    # Get the key (dtype) corresponding to the highest hierarchy
    resulting_dtype = next(
        key for key, value in dtype_hierarchy.items() if value == max_hierarchy
    )

    return resulting_dtype
