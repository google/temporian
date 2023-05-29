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

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.schema import Schema
from temporian.utils import string

# Maximum of printed index when calling repr(evset)
MAX_NUM_PRINTED_INDEX = 5

# Maximum of printed features when calling repr(evset)
MAX_NUM_PRINTED_FEATURES = 10

_PYTHON_DTYPE_MAPPING = {
    str: DType.STRING,
    # TODO: fix this, int doesn't have to be INT64 necessarily
    int: DType.INT64,
    np.int64: DType.INT64,
}

# Mapping of temporian types to and from numpy types.
#
# Remarks:
#   - np.object_ is not automatically converted into DType.STRING.
#   - Strings are always represented internally as np.str_ for features and str
#     for index values.
_DTYPE_MAPPING = {
    np.float64: DType.FLOAT64,
    np.float32: DType.FLOAT32,
    np.int64: DType.INT64,
    np.int32: DType.INT32,
    np.str_: DType.STRING,
    np.string_: DType.STRING,
    np.bool_: DType.BOOLEAN,
    np.datetime64: DType.INT64,
}
_DTYPE_REVERSE_MAPPING = {
    DType.FLOAT64: np.float64,
    DType.FLOAT32: np.float32,
    DType.INT64: np.int64,
    DType.INT32: np.int32,
    DType.STRING: np.str_,
    DType.BOOLEAN: np.bool_,
}


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
    raw_feature_values: Any,
) -> np.ndarray:
    """Normalies a list of feature values to temporian format.

    "normalize_features" should match "_DTYPE_MAPPING".
    """

    if not isinstance(raw_feature_values, np.ndarray):
        # The data is not a np.array

        if isinstance(raw_feature_values, list) and all(
            [isinstance(x, (str, bytes)) for x in raw_feature_values]
        ):
            # All the values are python strings.
            raw_feature_values = np.array(raw_feature_values, dtype=np.str_)
        else:
            raw_feature_values = np.array(raw_feature_values)

    else:
        if raw_feature_values.dtype.type == np.string_:
            raw_feature_values = raw_feature_values.astype(np.str_)

        if raw_feature_values.dtype.type == np.object_ and all(
            isinstance(x, str) for x in raw_feature_values
        ):
            # This is a np.array of python string.
            raw_feature_values = raw_feature_values.astype(np.str_)

        if raw_feature_values.dtype.type == np.datetime64:
            raw_feature_values = raw_feature_values.astype(
                "datetime64[s]"
            ).astype(np.int64)

    return raw_feature_values


def normalize_timestamps(
    raw_timestamps: Any,
) -> Tuple[np.ndarray, bool]:
    """Normalizes timestamps to temporian format.

    Returns:
        Normalized timestamps (numpy float64 of unix epoch in seconds) and if
        the raw timestamps look like a unix epoch.
    """

    # Convert to numpy array
    if not isinstance(raw_timestamps, np.ndarray):
        raw_timestamps = np.array(raw_timestamps)

    # raw_timestamps is represented as a number. Cast to float64.
    if raw_timestamps.dtype.type in [np.float32, np.int64, np.int32]:
        raw_timestamps = raw_timestamps.astype(np.float64)

    if raw_timestamps.dtype.type == np.float64:
        # Check NaN
        if np.isnan(raw_timestamps).any():
            raise ValueError("Timestamps contains NaN values.")

        return raw_timestamps, False

    if raw_timestamps.dtype.type == np.object_:
        if all(isinstance(x, str) for x in raw_timestamps):
            # raw_timestamps is a date. Cast to unix epoch in float64 seconds.
            raw_timestamps = (
                raw_timestamps.astype("datetime64[ns]").astype(np.float64) / 1e9
            )
            return raw_timestamps, True

        def is_pandas_timestamps(x):
            return (
                str(type(x))
                == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>"
            )

        if all(is_pandas_timestamps(x) for x in raw_timestamps):
            raw_timestamps = np.array([x.to_numpy() for x in raw_timestamps])

    if raw_timestamps.dtype.type == np.datetime64:
        # raw_timestamps is a date. Cast to unix epoch in float64 seconds.
        raw_timestamps = (
            raw_timestamps.astype("datetime64[ns]").astype(np.float64) / 1e9
        )
        return raw_timestamps, True

    object_description = (
        f"{raw_timestamps!r}.\nDetails: type={type(raw_timestamps)}"
    )
    if isinstance(raw_timestamps, np.ndarray):
        object_description += (
            f" np.dtype={raw_timestamps.dtype}"
            f" np.dtype.type={raw_timestamps.dtype.type}"
        )
        if raw_timestamps.dtype.type == np.object_:
            object_description += f" type(value[0])={type(raw_timestamps[0])}"

    raise ValueError(
        "Invalid timestamps value. Timestamps can be (1) a list / np.array of"
        " int,  float, int32, int64, float32, float64, str, np.datetime64, and"
        f" pd.Timestamp. Instead, got {object_description}"
    )


@dataclass
class IndexData:
    """Features and timestamps data for a single index item.

    Note: The "schema" constructor argument is only used for checking. If
    schema=None, no checking is done. Checking can be done manually with
    "index_data.check_schema(...)".

    Attributes:
        features: List of one-dimensional NumPy arrays representing the
            features.
        timestamps: One-dimensional NumPy array representing the timestamps.

    Example usage:
        ```
        >>> features = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> timestamps = np.array([0, 1, 2])
        >>> index_data = IndexData(features, timestamps)
        >>> len(index_data)
        3
        ```
    """

    features: List[np.ndarray]
    timestamps: np.ndarray

    def __init__(
        self,
        features: List[np.ndarray],
        timestamps: np.ndarray,
        schema: Optional[Schema],
    ) -> None:
        """Initializes the IndexData object by checking and setting the features
        and timestamps.

        Raises:
            ValueError: If features are not one-dimensional arrays.
            ValueError: If the number of elements in features and timestamps
                do not match.
        """

        self.features = features
        self.timestamps = timestamps

        if schema is not None:
            self.check_schema(schema)

    def check_schema(self, schema: Schema):
        # Check that the data (features & timestamps) matches the schema.

        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional arrays")

        if self.timestamps.dtype.type != np.float64:
            raise ValueError("Timestamps should be float64")

        if len(self.features) != len(schema.features):
            raise ValueError(
                "Wrong number of features. Event has "
                f"{len(self.features)} features while schema has "
                f"{len(schema.features)} features."
            )

        for feature_data, feature_schema in zip(self.features, schema.features):
            if feature_data.ndim != 1:
                raise ValueError("Features must be one-dimensional arrays")

            expected_numpy_type = _DTYPE_REVERSE_MAPPING[feature_schema.dtype]
            if feature_data.dtype.type != expected_numpy_type:
                raise ValueError(
                    "The schema does not match the feature dtype. Feature "
                    f"{feature_schema.name!r} has numpy dtype = "
                    f"{feature_data.dtype} but schema has temporian dtype = "
                    f"{feature_schema.dtype!r}. From the schema, the numpy"
                    f"type is expected to be {expected_numpy_type!r}."
                )

            if self.timestamps.shape != feature_data.shape:
                raise ValueError(
                    "The number of feature values for"
                    f" {feature_schema.name!r} ({len(feature_data)}) does not"
                    " match the number of timestamp values"
                    f" ({len(self.timestamps)})."
                )

    def __eq__(self, other) -> bool:
        if not isinstance(other, IndexData):
            return False

        if not np.array_equal(self.timestamps, other.timestamps):
            return False

        for f1, f2 in zip(self.features, other.features):
            if f1.dtype != f2.dtype:
                return False

            if f1.dtype.kind == "f":
                if not np.allclose(f1, f2, equal_nan=True):
                    return False
            else:
                if not np.array_equal(f1, f2):
                    return False

        return True

    def __len__(self) -> int:
        """Number of events / timesteps."""

        return len(self.timestamps)


class EventSet:
    """Actual temporal data.

    Use `tp.event_set` to create an event set manually.
    Use `tp.pd_dataframe_to_event_set` to create an event set from a pandas
    dataframe.
    """

    def __init__(
        self,
        data: Dict[Tuple, IndexData],
        schema: Schema,
        name: Optional[str] = None,
    ) -> None:
        self._data = data
        self._schema = schema
        self._name = name

        # Node created when "self.node()" is called.
        self._internal_node: Optional[Node] = None

    @property
    def data(self) -> Dict[Tuple, IndexData]:
        return self._data

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        self._name = name

    def get_arbitrary_index_value(self) -> Optional[Tuple]:
        """Gets an arbitrary index item.

        If the event set is empty, return None.
        """

        if self._data:
            return next(iter(self._data.keys()))
        return None

    def get_arbitrary_index_data(self) -> Optional[IndexData]:
        """Gets an arbitrary index item.

        If the event set is empty, return None.
        """

        if self._data:
            return next(iter(self._data.values()))
        return None

    def node(self, force_new_node=False) -> Node:
        print(
            "Warning: use `event_set.source_node()` instead of"
            " `event_set.node()`"
        )
        return self.source_node(force_new_node)

    def source_node(self, force_new_node=False) -> Node:
        """Creates a node able to consume the the event set.

        If called multiple times, always return the same node.
        Args:
            force_new_node: If false (default), return the same node if "node"
              is called multiple times. If true, return a new node each time.
        """

        if self._internal_node is not None and not force_new_node:
            # "node" was already called. Return the cached node.
            return self._internal_node

        self._internal_node = Node.create_with_new_reference(
            schema=self._schema,
            name=self._name,
        )
        return self._internal_node

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        index_names: Optional[List[str]] = None,
        timestamp_column: str = "timestamp",
        name: Optional[str] = None,
    ) -> EventSet:
        from temporian.implementation.numpy.data import io

        return io.pd_dataframe_to_event_set(
            df=df,
            index_names=index_names,
            timestamp_column=timestamp_column,
            name=name,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert a EventSet to a pandas DataFrame.

        Returns:
            DataFrame created from EventSet.
        """

        from temporian.implementation.numpy.data import io

        return io.event_set_to_pd_dataframe(self)

    def __repr__(self) -> str:
        def repr_features(features: List[np.ndarray]) -> str:
            """Repr for a list of features."""

            feature_repr = []
            for idx, (feature_schema, feature_data) in enumerate(
                zip(self.schema.features, features)
            ):
                if idx > MAX_NUM_PRINTED_FEATURES:
                    feature_repr.append("...")
                    break

                feature_repr.append(f"'{feature_schema.name}': {feature_data}")
            return "\n".join(feature_repr)

        # Representation of the "data" field
        with np.printoptions(precision=4, threshold=20):
            data_repr = []
            for i, (index_key, index_data) in enumerate(self.data.items()):
                if i > MAX_NUM_PRINTED_INDEX:
                    data_repr.append(f"... ({len(self.data) - i} remaining)")
                    break
                index_key_repr = []
                for index_value, index_name in zip(
                    index_key, self.schema.index_names()
                ):
                    index_key_repr.append(f"{index_name}={index_value}")
                index_key_repr = " ".join(index_key_repr)
                data_repr.append(
                    f"{index_key_repr} ({len(index_data.timestamps)} events):\n"
                    f"    timestamps: {index_data.timestamps}\n"
                    f"{string.indent(repr_features(index_data.features))}"
                )
            data_repr = string.indent("\n".join(data_repr))

        return (
            f"indexes: {self.schema.indexes}\n"
            f"features: {self.schema.features}\n"
            "events:\n"
            f"{data_repr}\n"
        )

    def __getitem__(self, index: Tuple) -> IndexData:
        return self.data[index]

    def __setitem__(self, index: Tuple, value: IndexData) -> None:
        self.data[index] = value

    def __eq__(self, other) -> bool:
        if not isinstance(other, EventSet):
            return False

        if self._name != other._name:
            return False

        if self._schema != other._schema:
            return False

        if self._data != other._data:
            return False

        return True

    def plot(self, *args, **wargs) -> Any:
        """Plots the event set. See tp.plot for details."""

        from temporian.implementation.numpy.data import plotter

        return plotter.plot(evsets=self, *args, **wargs)
