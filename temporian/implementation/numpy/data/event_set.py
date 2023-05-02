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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.utils import string

# Maximum of printed index when calling repr(evset)
MAX_NUM_PRINTED_INDEX = 5

# Maximum of printed features when calling repr(evset)
MAX_NUM_PRINTED_FEATURES = 10

PYTHON_DTYPE_MAPPING = {
    str: DType.STRING,
    # TODO: fix this, int doesn't have to be INT64 necessarily
    int: DType.INT64,
    np.int64: DType.INT64,
}

DTYPE_MAPPING = {
    np.float64: DType.FLOAT64,
    np.float32: DType.FLOAT32,
    np.int64: DType.INT64,
    np.int32: DType.INT32,
    np.str_: DType.STRING,
    np.string_: DType.STRING,
    np.bool_: DType.BOOLEAN,
}
DTYPE_REVERSE_MAPPING = {
    DType.FLOAT64: np.float64,
    DType.FLOAT32: np.float32,
    DType.INT64: np.int64,
    DType.INT32: np.int32,
    DType.STRING: np.str_,
    DType.BOOLEAN: np.bool_,
}


@dataclass
class IndexData:
    """Index data structure that holds features and timestamps.

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
    ) -> None:
        """Initializes the IndexData object by checking and setting the features
        and timestamps.

        Raises:
            ValueError: If features are not one-dimensional arrays.
            ValueError: If the number of elements in features and timestamps
                do not match.
        """
        shapes = [feature.shape for feature in features]
        if not all(len(shape) == 1 for shape in shapes):
            raise ValueError("Features must be one-dimensional arrays")

        if not all(shape == timestamps.shape for shape in shapes):
            raise ValueError(
                "Features must contain the same number of elements as the"
                " timestamp"
            )

        self.features = features
        self.timestamps = timestamps

    def __len__(self) -> int:
        """Returns the number of elements in the timestamps array."""
        return len(self.timestamps)


class EventSet:
    def __init__(
        self,
        data: Dict[Tuple, IndexData],
        feature_names: List[str],
        index_names: List[str],
        is_unix_timestamp: bool,
    ) -> None:
        self._data = data
        self._feature_names = feature_names
        self._index_names = index_names
        self._is_unix_timestamp = is_unix_timestamp

    @property
    def data(self) -> Dict[Tuple, IndexData]:
        return self._data

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def index_names(self) -> List[str]:
        return self._index_names

    @property
    def is_unix_timestamp(self) -> bool:
        return self._is_unix_timestamp

    # TODO: To remove
    @property
    def dtypes(self) -> Dict[str, DType]:
        return {
            feature_name: DTYPE_MAPPING[feature.dtype.type]
            for feature_name, feature in zip(
                self._feature_names, self.first_index_data().features
            )
        }

    # TODO: Rename to "dtypes".
    def dtypes_list(self) -> List[DType]:
        # TODO: Handle case where there is no data.
        return [feature.dtype for feature in self._first_index_features()]

    # TODO: Remove. Same as "len(self.feature_names)"
    @property
    def feature_count(self) -> int:
        return len(self._feature_names)

    # TODO: Remove. This is the same as "self.data".
    def iterindex(self) -> Iterable[Tuple[Tuple, IndexData]]:
        yield from self.data.items()

    # TODO: improve numpy backend index handling
    def index_dtypes(self) -> Dict[str, DType]:
        return (
            {
                index_name: PYTHON_DTYPE_MAPPING[type(index_key)]
                for index_name, index_key in zip(
                    self._index_names, self.first_index_key()
                )
            }
            if self._data
            else {}
        )

    def first_index_key(self) -> Optional[Tuple]:
        if self._data is None or len(self._data) == 0:
            return None

        return next(iter(self._data))

    # TODO: Remove.
    def first_index_data(self) -> IndexData:
        if self.first_index_key() is None:
            return []
        return self[self.first_index_key()]

    # TODO: Remove.
    def _first_index_features(self) -> List[np.ndarray]:
        return list(self._data.values())[0].features

    # TODO: Do not recompute the schema on the fly. Instead, keep an internal
    # Event / Node. This Node is possibly given as constructor argument.
    def node(self) -> Node:
        return Node(
            features=[
                Feature(name, dtype) for name, dtype in self.dtypes.items()
            ],
            sampling=Sampling(
                index_levels=[
                    (index_name, index_dtype)
                    for index_name, index_dtype in self.index_dtypes().items()
                ],
                is_unix_timestamp=self._is_unix_timestamp,
            ),
        )

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        index_names: Optional[List[str]] = None,
        timestamp_column: str = "timestamp",
        is_sorted: bool = False,
    ) -> EventSet:
        """Creates an EventSet from a pandas DataFrame.

        Args:
            df: DataFrame to convert to an EventSet.
            index_names: Names of the DataFrame columns to be used as index for
                the event set. Defaults to [].
            timestamp_column: Name of the column containing the timestamps.
                Supported date types:
                `{np.datetime64, pd.Timestamp, datetime.datetime}`.
                Timestamps of these types are converted to UTC epoch float.
            is_sorted: If True, the DataFrame is assumed to be sorted by
                timestamp. If False, the DataFrame will be sorted by timestamp.


        Returns:
            EventSet created from DataFrame.

        Raises:
            ValueError: If `index_names` or `timestamp_column` are not in `df`'s
                columns.
            ValueError: If a column has an unsupported dtype.

        Example:
            >>> import pandas as pd
            >>> from temporian.implementation.numpy.data.event_set import EventSet
            >>> df = pd.DataFrame(
            ...     data=[
            ...         [666964, 1.0, 740.0],
            ...         [666964, 2.0, 508.0],
            ...         [574016, 3.0, 573.0],
            ...     ],
            ...     columns=["product_id", "timestamp", "costs"],
            ... )
            >>> evset = EventSet.from_dataframe(df, index_names=["product_id"])
        """

        def convert_timestamp_column_to_unix_epoch_float(
            timestamp_column: pd.Series,
        ) -> pd.DataFrame:
            """Converts a timestamp column to Unix Epoch Float.

            Args:
                timestamp_column: Timestamp column to convert.

            Returns:
                Timestamp column converted to Unix Epoch float.
            """
            # check if timestamp column contains missing values and raise error
            if timestamp_column.isna().any():
                raise ValueError(
                    f"Cannot convert timestamp column {timestamp_column.name} "
                    "to Unix Epoch Float because it contains missing values."
                )

            # if timestamp_column is already float64, ignore it
            if timestamp_column.dtype == "float64":
                return timestamp_column

            # if timestamp_column is int or float != float64 convert to float64
            if timestamp_column.dtype.kind in ("i", "f"):
                return timestamp_column.astype("float64")

            # string and objects will be converted to datetime, then to float
            timestamp_column = pd.to_datetime(timestamp_column, errors="raise")
            timestamp_column = timestamp_column.view("int64") / 1e9
            return timestamp_column

        df = df.copy(deep=False)
        if index_names is None:
            index_names = []

        # check index names and timestamp name are in df columns
        missing_columns = [
            column
            for column in index_names + [timestamp_column]
            if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing columns {missing_columns} in DataFrame. "
                f"Columns: {df.columns}"
            )

        # check timestamp_column is not on index_names
        if timestamp_column in index_names:
            raise ValueError(
                f"Timestamp column {timestamp_column} cannot be on index_names"
            )

        # check if created sampling's values will be unix timestamps
        is_unix_timestamp = df[timestamp_column].dtype.kind not in ("i", "f")

        # convert timestamp column to Unix Epoch Float
        df[timestamp_column] = convert_timestamp_column_to_unix_epoch_float(
            df[timestamp_column]
        )

        # sort by timestamp if it's not sorted
        # TODO: we may consider using kind="mergesort" if we know that most of
        # the time the data will be sorted.
        if not is_sorted and not np.all(np.diff(df[timestamp_column]) >= 0):
            df = df.sort_values(by=timestamp_column)

        # check column dtypes, every dtype should be a key of DTYPE_MAPPING
        for column in df.columns:
            # if dtype is object, check if it only contains string values
            if df[column].dtype.type is np.object_:
                df[column] = df[column].fillna("")
                # Check if there are any non-string elements in the column
                non_string_mask = df[column].map(type) != str
                if non_string_mask.any():
                    raise ValueError(
                        f'Cannot convert column "{column}". Column of type'
                        ' "Object" can only have string values. However, the'
                        " following non-string values were found: "
                        f" {df[column][non_string_mask]}"
                    )
                # convert object column to np.string_
                df[column] = df[column].astype("string")

            # convert pandas' StringDtype to np.string_
            elif df[column].dtype.type is np.string_:
                df[column] = df[column].str.decode("utf-8").astype("string")

            elif (
                df[column].dtype.type not in DTYPE_MAPPING
                and df[column].dtype.type is not str
            ):
                raise ValueError(
                    f"Unsupported dtype {df[column].dtype} for column"
                    f" {column}. Supported dtypes: {DTYPE_MAPPING.keys()}"
                )

        # columns that are not indexes or timestamp
        feature_names = [
            column
            for column in df.columns
            if column not in index_names + [timestamp_column]
        ]
        # fill missing values with np.nan
        df = df.fillna(np.nan)

        data = {}
        if index_names:
            grouping_key = (
                index_names[0] if len(index_names) == 1 else index_names
            )
            group_by_indexes = df.groupby(grouping_key)

            for index, group in group_by_indexes:
                timestamps = group[timestamp_column].to_numpy()

                # Convert group to tuple, useful when its only one value
                if not isinstance(index, tuple):
                    index = (index,)

                data[index] = IndexData(
                    features=[
                        group[feature_name].to_numpy(
                            dtype=group[feature_name].dtype.type
                        )
                        for feature_name in feature_names
                    ],
                    timestamps=timestamps,
                )

        # user did not provide an index
        else:
            timestamps = df[timestamp_column].to_numpy()
            data[()] = IndexData(
                features=[
                    df[feature_name].to_numpy(dtype=df[feature_name].dtype.type)
                    for feature_name in feature_names
                ],
                timestamps=timestamps,
            )

        return EventSet(
            data=data,
            feature_names=feature_names,
            index_names=index_names,
            is_unix_timestamp=is_unix_timestamp,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert a EventSet to a pandas DataFrame.

        Returns:
            DataFrame created from EventSet.
        """
        column_names = self._index_names + self._feature_names + ["timestamp"]
        data = {column_name: [] for column_name in column_names}

        for index_key, index_data in self._data.items():
            # parse timestamps. TODO: add output dtypes argument
            data["timestamp"].extend(index_data.timestamps.astype(np.float64))

            # parse feature data
            for feature_name, feature in zip(
                self._feature_names, index_data.features
            ):
                data[feature_name].extend(feature)

            # TODO: do we need this?
            if not isinstance(index_key, tuple):
                index_key = (index_key,)

            # parse index values as columns
            for i, index_name in enumerate(self._index_names):
                data[index_name].extend(
                    [index_key[i]] * len(index_data.timestamps)
                )

        # convert dictionary to pandas DataFrame
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        def repr_features(features: List[np.ndarray]) -> str:
            """Repr for a list of features."""

            feature_repr = []
            for idx, (feature_name, feature) in enumerate(
                zip(self.feature_names, features)
            ):
                if idx > MAX_NUM_PRINTED_FEATURES:
                    feature_repr.append("...")
                    break

                feature_repr.append(
                    f"{feature_name}<{feature.dtype}>: {feature})"
                )
            return "\n".join(feature_repr)

        # Representation of the "data" field
        with np.printoptions(precision=4, threshold=20):
            data_repr = []
            for i, (index_key, index_data) in enumerate(self._data.items()):
                if i > MAX_NUM_PRINTED_INDEX:
                    data_repr.append("...")
                    break
                data_repr.append(
                    f"{index_key}:"
                    f" {index_data.timestamps}\n{string.indent(repr_features(index_data.features))}"
                )
            data_repr = string.indent("\n".join(data_repr))

        return (
            "data:"
            f"\n\t\tindex_names={self.index_names}"
            f"\n\t\tfeature_names={self.feature_names}"
            f"\n{data_repr}"
        )

    def __getitem__(self, index: Tuple) -> IndexData:
        return self.data[index]

    def __setitem__(self, index: Tuple, value: IndexData) -> None:
        self.data[index] = value

    def __eq__(self, __o: object) -> bool:
        # tolerance levels for float comparison. TODO: move to appropiate place

        # TODO: Remove. Equality tests should be exact. Create a "Near" function
        # if necessary.
        rtol = 1e-9
        atol = 1e-9

        if not isinstance(__o, EventSet):
            return False

        # check same features
        if self._feature_names != __o.feature_names:
            return False

        # check same index
        if self._index_names != __o.index_names:
            return False

        # check unix timestamp
        if self._is_unix_timestamp != __o.is_unix_timestamp:
            return False

        # check same data
        if len(self._data) != len(__o.data):
            return False

        for index_key, index_data_self in self._data.items():
            if index_key not in __o.data:
                return False

            # check same features
            index_data_other = __o[index_key]
            for feature_self, feature_other in zip(
                index_data_self.features, index_data_other.features
            ):
                if feature_self.dtype.type != feature_other.dtype.type:
                    return False

                # check if the array has a float dtype. If so, compare with
                # `allclose`
                if feature_self.dtype.kind == "f":
                    equal = np.allclose(
                        feature_self,
                        feature_other,
                        rtol=rtol,
                        atol=atol,
                        equal_nan=True,
                    )
                else:
                    # compare non-float arrays
                    equal = np.array_equal(feature_self, feature_other)

                if not equal:
                    return False

            # check same timestamps
            if not np.allclose(
                index_data_self.timestamps,
                index_data_other.timestamps,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                return False

        return True

    def plot(self, *args, **wargs) -> Any:
        """Plots the event set. See tp.plot for details."""

        from temporian.implementation.numpy.data import plotter

        return plotter.plot(evsets=self, *args, **wargs)
