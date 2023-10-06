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
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from temporian.core.data.dtype import DType
from temporian.core.data.duration_utils import datetime64_array_to_float64
from temporian.core.data.node import EventSetNode

if TYPE_CHECKING:
    from temporian.core.typing import (
        IndexKey,
        TargetDtypes,
        IndexKeyItem,
        NormalizedIndexKey,
        NormalizedIndexKeyItem,
        IndexKeyList,
    )

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


def normalize_index_item(x: IndexKeyItem) -> NormalizedIndexKeyItem:
    if isinstance(x, str):
        return x.encode()
    elif isinstance(x, (int, str, bytes)):
        return x
    raise ValueError(f"Non supported index item {x}")


def normalize_index_key(index: Optional[IndexKey]) -> NormalizedIndexKey:
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

    def _str_to_bytes(feat_array: np.ndarray) -> np.ndarray:
        """Encode string/object/bytes to np.bytes, using UTF-8 encoding"""
        return np.char.encode(feat_array, "UTF-8")

    # Convert pandas, list, tuples -> np.ndarray
    if str(type(feature_values)) == "<class 'pandas.core.series.Series'>":
        if feature_values.dtype == "object":
            feature_values = feature_values.fillna("")
        feature_values = feature_values.to_numpy(copy=True)
    elif isinstance(feature_values, (tuple, list)):
        # Convert list/tuple to array
        feature_values = np.array(feature_values)
    elif not isinstance(feature_values, np.ndarray):
        raise ValueError(
            "Feature values should be provided in a tuple, list, numpy array or"
            f" pandas Series. Got type {type(feature_values)} instead."
        )

    # Next steps: Assume np.ndarray, normalize dtype
    assert isinstance(feature_values, np.ndarray)

    array_dtype = feature_values.dtype.type

    # Convert np.datetime -> np.float64
    if array_dtype == np.datetime64:
        # nanosecond resolution as in timestamps
        feature_values = datetime64_array_to_float64(feature_values)

    # Convert np.object_, np.str_ -> np.bytes_
    elif array_dtype == np.str_:
        feature_values = _str_to_bytes(feature_values)
    elif array_dtype == np.object_:
        logging.warning(
            (
                'Feature "%s" is an array of numpy.object_ and will be'
                " casted to numpy.string_ (Note: numpy.string_ is"
                " equivalent to numpy.bytes_)."
            ),
            name,
        )
        feature_values = _str_to_bytes(feature_values.astype(str, copy=False))

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

    # values is represented as a number. Copy and cast to float64.
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(
        values.dtype, np.floating
    ):
        values = values.astype(np.float64, copy=True)

    if values.dtype.type == np.float64:
        # Check NaN
        if np.isnan(values).any():
            raise ValueError("Timestamps contain NaN values.")

        return values, False

    if values.dtype.type in [np.str_, np.bytes_, np.object_]:
        # Raises ValueError if cannot parse a value
        values = values.astype("datetime64[ns]")

    if values.dtype.type == np.datetime64:
        # values is a date. Cast to unix epoch in float64 seconds.
        values = datetime64_array_to_float64(values)
        return values, True

    raise ValueError(
        f"Invalid timestamps array dtype={values.dtype}."
        " Supported types are: integers, floating point, strings or objects."
    )


def normalize_index_key_list(
    indexes: Optional[IndexKeyList],
    available_indexes: Optional[List[IndexKey]] = None,
) -> List[NormalizedIndexKey]:
    """Normalizes a list of index keys.

    If `indexes` is None: if available_indexes is not None it returns those,
    else returns an empty list."""

    if indexes is None:
        if available_indexes is not None:
            # All available indexes
            normalized_indexes = available_indexes
        else:
            normalized_indexes = []

    elif isinstance(indexes, list):
        # e.g. indexes=["a", ("b",)]
        normalized_indexes = [
            v if isinstance(v, tuple) else (v,) for v in indexes
        ]

    elif isinstance(indexes, tuple):
        # e.g. indexes=("a",)
        normalized_indexes = [indexes]

    else:
        # e.g. indexes="a"
        normalized_indexes = [(indexes,)]

    normalized_indexes = [normalize_index_key(x) for x in normalized_indexes]

    return normalized_indexes


def normalize_dtype(x: Any) -> DType:
    if isinstance(x, DType):
        return x
    if x == int:
        return DType.INT64
    if x == float:
        return DType.FLOAT64
    if x == str:
        return DType.STRING
    if x == bool:
        return DType.BOOLEAN
    raise ValueError(f"Cannot normalize {x!r} as a DType.")


def normalize_target_dtypes(
    input: EventSetNode,
    target: TargetDtypes,
) -> Tuple[
    Optional[DType],
    Optional[Dict[str, DType]],
    Optional[Dict[DType, DType]],
]:
    # Convert 'target' to one of these:
    dtype: Optional[DType] = None
    feature_name_to_dtype: Optional[Dict[str, DType]] = None
    dtype_to_dtype: Optional[Dict[DType, DType]] = None

    # Further type verifications are done in the operator
    if isinstance(target, dict):
        keys_are_strs = all(isinstance(v, str) for v in target.keys())
        keys_are_dtypes = all(
            isinstance(v, (DType, type)) for v in target.keys()
        )
        values_are_dtypes = all(
            isinstance(v, (DType, type)) for v in target.values()
        )

        if keys_are_strs and values_are_dtypes:
            feature_name_to_dtype = {
                key: normalize_dtype(value) for key, value in target.items()
            }

            input_feature_names = input.schema.feature_names()
            for feature_name in feature_name_to_dtype.keys():
                if feature_name not in input_feature_names:
                    raise ValueError(f"Unknown feature {feature_name!r}")

        elif keys_are_dtypes and values_are_dtypes:
            dtype_to_dtype = {
                normalize_dtype(key): normalize_dtype(value)
                for key, value in target.items()
            }
    elif isinstance(target, DType) or target in [float, int, str, bool]:
        dtype = normalize_dtype(target)

    if (
        dtype is None
        and feature_name_to_dtype is None
        and dtype_to_dtype is None
    ):
        raise ValueError(
            "`target` should be one of the following: (1) a Temporian dtype"
            " e.g. tp.float64, (2) a dictionary of feature name (str) to"
            " temporian dtype, or (3) a dictionary of temporian dtype to"
            " temporian dtype. Alternatively, Temporian dtypes can be replaced"
            " with python type. For example cast(..., target=float) is"
            " equivalent to cast(..., target=tp.float64).\nInstead got,"
            f" `target` = {target!r}."
        )

    return dtype, feature_name_to_dtype, dtype_to_dtype


def build_dtypes_list_from_target_dtypes(
    input: EventSetNode,
    dtype: Optional[DType] = None,
    dtype_to_dtype: Optional[Dict[DType, DType]] = None,
    feature_name_to_dtype: Optional[Dict[str, DType]] = None,
) -> List[DType]:
    """Builds a list of output dtypes for the input based on the output of
    normalize_target_dtypes."""

    if dtype is not None:
        return [dtype] * len(input.schema.features)

    if feature_name_to_dtype is not None:
        return [
            feature_name_to_dtype.get(f.name, f.dtype)
            for f in input.schema.features
        ]

    if dtype_to_dtype is not None:
        return [
            dtype_to_dtype.get(f.dtype, f.dtype) for f in input.schema.features
        ]

    # If none were set, return the input dtypes
    return [f.dtype for f in input.schema.features]
