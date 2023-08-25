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

"""Shared data type normalization functions."""

from __future__ import annotations
import datetime
import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np

from temporian.core.data.dtype import DType

if TYPE_CHECKING:
    from temporian.core.typing import IndexKey, IndexKeyItem

# Mapping of temporian types to and from numpy types.
#
# Remarks:
#   - np.object_ is not automatically converted into DType.STRING.
#   - Strings are always represented internally as np.bytes_ for features and
#       bytes for index groups.
_DTYPE_MAPPING = {
    np.float64: DType.FLOAT64,
    np.float32: DType.FLOAT32,
    np.int64: DType.INT64,
    np.int32: DType.INT32,
    np.str_: DType.STRING,
    np.bytes_: DType.STRING,
    np.bool_: DType.BOOLEAN,
    np.datetime64: DType.INT64,
}
_DTYPE_REVERSE_MAPPING = {
    DType.FLOAT64: np.float64,
    DType.FLOAT32: np.float32,
    DType.INT64: np.int64,
    DType.INT32: np.int32,
    DType.STRING: np.bytes_,
    DType.BOOLEAN: np.bool_,
}


def normalize_index_item(x: IndexKeyItem) -> IndexKeyItem:
    if isinstance(x, str):
        return x.encode()
    elif isinstance(x, (int, str, bytes)):
        return x
    raise ValueError(f"Non supported index item {x}")


def normalize_index_key(
    index: Optional[Union[IndexKeyItem, IndexKey]]
) -> IndexKey:
    if index is None:
        return tuple()
    if isinstance(index, tuple):
        return tuple([normalize_index_item(x) for x in index])
    return (normalize_index_item(index),)


def is_supported_numpy_dtype(numpy_dtype) -> bool:
    return numpy_dtype in _DTYPE_MAPPING


def numpy_dtype_to_tp_dtype(feature_name: str, numpy_dtype) -> DType:
    """Converts a numpy dtype into a temporian dtype."""

    if numpy_dtype not in _DTYPE_MAPPING:
        raise ValueError(
            f"Features {feature_name!r} with dtype {numpy_dtype} cannot be"
            " imported in Temporian. Supported"
            f" dtypes={list(_DTYPE_MAPPING.keys())}."
        )

    return _DTYPE_MAPPING[numpy_dtype]


def numpy_array_to_tp_dtype(
    feature_name: str, numpy_array: np.ndarray
) -> DType:
    """Gets the matching temporian dtype of a numpy array."""

    return numpy_dtype_to_tp_dtype(feature_name, numpy_array.dtype.type)


def tp_dtype_to_np_dtype(dtype: DType) -> Any:
    return _DTYPE_REVERSE_MAPPING[dtype]


def normalize_features(
    feature_values: Any,
    name: str,
) -> np.ndarray:
    """Normalizes a list of feature values to temporian format.

    Keep this function in sync with the documentation of "io.event_set".

    `normalize_features` should match `_DTYPE_MAPPING`.
    """

    if str(type(feature_values)) == "<class 'pandas.core.series.Series'>":
        if feature_values.dtype == "object":
            feature_values = feature_values.fillna("")
        feature_values = feature_values.to_numpy(copy=True)
    elif isinstance(feature_values, (tuple, list)):
        # Convert list/tuple to array

        # Looks for an indication of a string or non-string array.
        is_string = False
        for x in feature_values:
            if isinstance(x, (str, bytes)):
                is_string = True
                break
            if isinstance(x, (int, bool, float)):
                is_string = False
                break

        if is_string:
            # All the values are python strings.
            feature_values = np.array(feature_values, dtype=np.bytes_)
        else:
            feature_values = np.array(feature_values)
    elif not isinstance(feature_values, np.ndarray):
        raise ValueError(
            "Feature values should be provided in a tuple, list, numpy array or"
            f" pandas Series. Got type {type(feature_values)} instead."
        )

    if feature_values.dtype.type == np.string_:
        feature_values = feature_values.astype(np.bytes_)

    if feature_values.dtype.type == np.object_:
        logging.warning(
            (
                'Feature "%s" is an array of numpy.object_ and was casted to'
                " numpy.string_ (Note: numpy.string_ is equivalent to"
                " numpy.bytes_)."
            ),
            name,
        )
        feature_values = feature_values.astype(np.bytes_)

    if feature_values.dtype.type == np.datetime64:
        feature_values = feature_values.astype("datetime64[s]").astype(np.int64)

    return feature_values


def normalize_timestamps(
    values: Any,
) -> Tuple[np.ndarray, bool]:
    """Normalizes timestamps to temporian format.

    Keep this function in sync with the documentation of "io.event_set".

    Returns:
        Normalized timestamps (numpy float64 of unix epoch in seconds) and if
        the raw timestamps look like a unix epoch.
    """

    # Convert to numpy array
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    # values is represented as a number. Cast to float64.
    if values.dtype.type in [np.float32, np.int64, np.int32]:
        values = values.astype(np.float64)

    if values.dtype.type == np.float64:
        # Check NaN
        if np.isnan(values).any():
            raise ValueError("Timestamps contains NaN values.")

        return values, False

    if values.dtype.type in [np.str_, np.bytes_]:
        values = values.astype("datetime64[ns]")

    if values.dtype.type == np.object_:
        if all(isinstance(x, str) for x in values) or all(
            isinstance(x, datetime.datetime) for x in values
        ):
            # values is a date. Cast to unix epoch in float64 seconds.
            values = values.astype("datetime64[ns]")

        elif all(
            str(type(x)) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>"
            for x in values
        ):
            values = np.array([x.to_numpy() for x in values])

    if values.dtype.type == np.datetime64:
        # values is a date. Cast to unix epoch in float64 seconds.
        values = values.astype("datetime64[ns]").astype(np.float64) / 1e9
        return values, True

    object_description = f"{values!r}.\nDetails: type={type(values)}"
    if isinstance(values, np.ndarray):
        object_description += (
            f" np.dtype={values.dtype} np.dtype.type={values.dtype.type}"
        )
        if values.dtype.type == np.object_:
            object_description += f" type(value[0])={type(values[0])}"

    # Keep this function in sync with the documentation of "io.event_set".
    raise ValueError(
        "Invalid timestamps value. Check `tp.event_set` documentation for the"
        " list of supported timestamp types. Instead, got"
        f" {object_description}."
    )
