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

# Mapping of temporian types to / from numpy types.
#
# np.object_ is not automatically converted into DType.STRING
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
    DType.STRING: np.string_,
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
            raw_feature_values = np.array(raw_feature_values, dtype=np.string_)
        else:
            raw_feature_values = np.array(raw_feature_values)

    else:
        if raw_feature_values.dtype.type == np.str_:
            raw_feature_values = raw_feature_values.astype(np.string_)

        if raw_feature_values.dtype.type == np.object_ and all(
            isinstance(x, str) for x in raw_feature_values
        ):
            # This is a np.array of python string.
            raw_feature_values = raw_feature_values.astype(np.string_)

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

    if raw_timestamps.dtype.type == np.datetime64:
        # raw_timestamps is a date. Cast to unix epoch in float64 seconds.
        raw_timestamps = (
            raw_timestamps.astype("datetime64[ns]").astype(np.float64) / 1e9
        )
        return raw_timestamps, True

    if raw_timestamps.dtype.type == np.object_ and all(
        isinstance(x, str) for x in raw_timestamps
    ):
        # raw_timestamps is a date. Cast to unix epoch in float64 seconds.
        raw_timestamps = (
            raw_timestamps.astype("datetime64[ns]").astype(np.float64) / 1e9
        )
        return raw_timestamps, True

    raise ValueError(f"No support values for timestamps: {raw_timestamps}")


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
                "Wrong number of features. Event has"
                f" {len(self.features)} features while schema indicates"
                f" {len(schema.features)} features."
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
                    "The number of feature values does not match the number of"
                    " timestamps."
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
    Use `tp.EventSet.from_dataframe` to create an event set from a pandas
    dataframe.

    TODO: tp.EventSet.from_dataframe -> tp.pd_dataframe_to_event_set
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

    def node(self, force_new_node=False) -> Node:
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
        is_sorted: bool = False,
        name: Optional[str] = None,
    ) -> EventSet:
        from temporian.implementation.numpy.data import io

        return io.pd_dataframe_to_event_set(
            df=df,
            index_names=index_names,
            timestamp_column=timestamp_column,
            name=name,
        )
        # """Creates an EventSet from a pandas DataFrame.

        # Args:
        #     df: DataFrame to convert to an EventSet.
        #     index_names: Names of the DataFrame columns to be used as index for
        #         the event set. Defaults to [].
        #     timestamp_column: Name of the column containing the timestamps.
        #         Supported date types:
        #         `{np.datetime64, pd.Timestamp, datetime.datetime}`.
        #         Timestamps of these types are converted to UTC epoch float.
        #     is_sorted: If True, the DataFrame is assumed to be sorted by
        #         timestamp. If False, the DataFrame will be sorted by timestamp.
        #     name: Optional name for the EventSet.

        # Returns:
        #     EventSet created from DataFrame.

        # Raises:
        #     ValueError: If `index_names` or `timestamp_column` are not in `df`'s
        #         columns.
        #     ValueError: If a column has an unsupported dtype.

        # Example:
        #     >>> import pandas as pd
        #     >>> from temporian.implementation.numpy.data.event_set import EventSet
        #     >>> df = pd.DataFrame(
        #     ...     data=[
        #     ...         [666964, 1.0, 740.0],
        #     ...         [666964, 2.0, 508.0],
        #     ...         [574016, 3.0, 573.0],
        #     ...     ],
        #     ...     columns=["product_id", "timestamp", "costs"],
        #     ... )
        #     >>> evset = EventSet.from_dataframe(df, index_names=["product_id"])
        # """

        # # TODO: Detect "is_sorted" automatically.

        # df = df.copy(deep=False)
        # if index_names is None:
        #     index_names = []

        # # check index names and timestamp name are in df columns
        # missing_columns = [
        #     column
        #     for column in index_names + [timestamp_column]
        #     if column not in df.columns
        # ]
        # if missing_columns:
        #     raise ValueError(
        #         f"Missing columns {missing_columns} in DataFrame. "
        #         f"Columns: {df.columns}"
        #     )

        # # check timestamp_column is not on index_names
        # if timestamp_column in index_names:
        #     raise ValueError(
        #         f"Timestamp column {timestamp_column} cannot be on index_names"
        #     )

        # # check if created sampling's values will be unix timestamps
        # is_unix_timestamp = df[timestamp_column].dtype.kind not in ("i", "f")

        # # convert timestamp column to Unix Epoch Float
        # df[timestamp_column] = normalize_timestamps(df[timestamp_column])

        # # sort by timestamp if it's not sorted
        # # TODO: we may consider using kind="mergesort" if we know that most of
        # # the time the data will be sorted.
        # if not is_sorted and not np.all(np.diff(df[timestamp_column]) >= 0):
        #     df = df.sort_values(by=timestamp_column)

        # # check column dtypes, every dtype should be a key of DTYPE_MAPPING
        # for column in df.columns:
        #     # if dtype is object, check if it only contains string values
        #     if df[column].dtype.type is np.object_:
        #         df[column] = df[column].fillna("")
        #         # Check if there are any non-string elements in the column
        #         non_string_mask = df[column].map(type) != str
        #         if non_string_mask.any():
        #             raise ValueError(
        #                 f'Cannot convert column "{column}". Column of type'
        #                 ' "Object" can only have string values. However, the'
        #                 " following non-string values were found: "
        #                 f" {df[column][non_string_mask]}"
        #             )
        #         # convert object column to np.string_
        #         df[column] = df[column].astype("string")

        #     # convert pandas' StringDtype to np.string_
        #     elif df[column].dtype.type is np.string_:
        #         df[column] = df[column].str.decode("utf-8").astype("string")

        #     elif (
        #         df[column].dtype.type not in DTYPE_MAPPING
        #         and df[column].dtype.type is not str
        #     ):
        #         raise ValueError(
        #             f"Unsupported dtype {df[column].dtype} for column"
        #             f" {column}. Supported dtypes: {DTYPE_MAPPING.keys()}"
        #         )

        # # columns that are not indexes or timestamp
        # feature_names = [
        #     column
        #     for column in df.columns
        #     if column not in index_names + [timestamp_column]
        # ]
        # # fill missing values with np.nan
        # df = df.fillna(np.nan)

        # data = {}
        # if index_names:
        #     grouping_key = (
        #         index_names[0] if len(index_names) == 1 else index_names
        #     )
        #     group_by_indexes = df.groupby(grouping_key)

        #     for index, group in group_by_indexes:
        #         timestamps = group[timestamp_column].to_numpy()

        #         # Convert group to tuple, useful when its only one value
        #         if not isinstance(index, tuple):
        #             index = (index,)

        #         data[index] = IndexData(
        #             features=[
        #                 group[feature_name].to_numpy(
        #                     dtype=group[feature_name].dtype.type
        #                 )
        #                 for feature_name in feature_names
        #             ],
        #             timestamps=timestamps,
        #         )

        # # user did not provide an index
        # else:
        #     timestamps = df[timestamp_column].to_numpy()
        #     data[()] = IndexData(
        #         features=[
        #             df[feature_name].to_numpy(dtype=df[feature_name].dtype.type)
        #             for feature_name in feature_names
        #         ],
        #         timestamps=timestamps,
        #     )

        # return EventSet(
        #     data=data,
        #     feature_names=feature_names,
        #     index_names=index_names,
        #     is_unix_timestamp=is_unix_timestamp,
        #     name=name,
        # )

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

        # if len(self._data) != len(other.data):
        #     return False

        # for index_key, index_data_self in self._data.items():
        #     if index_key not in other.data:
        #         return False

        #     # check same features
        #     index_data_other = other[index_key]
        #     for feature_self, feature_other in zip(
        #         index_data_self.features, index_data_other.features
        #     ):
        #         if feature_self.dtype.type != feature_other.dtype.type:
        #             return False

        #         # check if the array has a float dtype. If so, compare with
        #         # `allclose`
        #         if feature_self.dtype.kind == "f":
        #             equal = np.allclose(
        #                 feature_self,
        #                 feature_other,
        #                 rtol=rtol,
        #                 atol=atol,
        #                 equal_nan=True,
        #             )
        #         else:
        #             # compare non-float arrays
        #             equal = np.array_equal(feature_self, feature_other)

        #         if not equal:
        #             return False

        #     # check same timestamps
        #     if not np.allclose(
        #         index_data_self.timestamps,
        #         index_data_other.timestamps,
        #         rtol=rtol,
        #         atol=atol,
        #         equal_nan=True,
        #     ):
        #         return False

        # return True

    def plot(self, *args, **wargs) -> Any:
        """Plots the event set. See tp.plot for details."""

        from temporian.implementation.numpy.data import plotter

        return plotter.plot(evsets=self, *args, **wargs)


# def _convert_timestamp_column_to_unix_epoch_float(
#     timestamp_column: pd.Series,
# ) -> pd.DataFrame:
#     """Converts a timestamp column to Unix Epoch Float.

#     Args:
#         timestamp_column: Timestamp column to convert.

#     Returns:
#         Timestamp column converted to Unix Epoch float.
#     """
#     # check if timestamp column contains missing values and raise error
#     if timestamp_column.isna().any():
#         raise ValueError(
#             f"Cannot convert timestamp column {timestamp_column.name} "
#             "to Unix Epoch Float because it contains missing values."
#         )

#     # if timestamp_column is already float64, ignore it
#     if timestamp_column.dtype == "float64":
#         return timestamp_column

#     # if timestamp_column is int or float != float64 convert to float64
#     if timestamp_column.dtype.kind in ("i", "f"):
#         return timestamp_column.astype("float64")

#     # string and objects will be converted to datetime, then to float
#     timestamp_column = pd.to_datetime(timestamp_column, errors="raise")
#     timestamp_column = timestamp_column.view("int64") / 1e9
#     return timestamp_column
