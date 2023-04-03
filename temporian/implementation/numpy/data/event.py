from typing import Any, Dict, List, Tuple, Sequence

import numpy as np
import pandas as pd

from temporian.core.data import dtype
from temporian.core.data.duration import convert_date_to_duration
from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.utils import string

DTYPE_MAPPING = {
    np.float64: dtype.FLOAT64,
    np.float32: dtype.FLOAT32,
    np.int64: dtype.INT64,
    np.int32: dtype.INT32,
}

DTYPE_REVERSE_MAPPING = {v: k for k, v in DTYPE_MAPPING.items()}
DTYPE_REVERSE_MAPPING[dtype.STRING] = np.str_

# Maximum of printed index when calling repr(event)
MAX_NUM_PRINTED_INDEX = 5

# Maximum of printed features when calling repr(event)
MAX_NUM_PRINTED_FEATURES = 10


def dtype_to_np_dtype(src: dtype.DType) -> Any:
    return DTYPE_REVERSE_MAPPING[src]


class NumpyFeature:
    def __init__(self, name: str, data: np.ndarray) -> None:
        if len(data.shape) > 1:
            raise ValueError(
                "NumpyFeatures can only be created from flat arrays. Passed"
                f" input's shape: {len(data.shape)}"
            )
        if data.dtype.type is np.str_ or data.dtype.type is np.string_:
            self.dtype: dtype.DType = dtype.STRING
        else:
            if data.dtype.type not in DTYPE_MAPPING:
                raise ValueError(
                    f"Unsupported dtype {data.dtype} for NumpyFeature: {name}."
                    f" Supported dtypes: {DTYPE_MAPPING.keys()}, np.str_ and "
                    "np.string_"
                )
            self.dtype: dtype.DType = DTYPE_MAPPING[data.dtype.type]

        self.name = name
        self.data = data

    def __repr__(self) -> str:
        return f"{self.name}: {self.data.__repr__()}"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, NumpyFeature):
            return False

        if self.name != __o.name:
            return False

        if self.dtype == dtype.STRING:
            return np.array_equal(self.data, __o.data)

        return np.array_equal(self.data, __o.data, equal_nan=True)

    def schema(self) -> Feature:
        return Feature(self.name, self.dtype)


class NumpyEvent:
    def __init__(
        self,
        data: Dict[Tuple, List[NumpyFeature]],
        sampling: NumpySampling,
    ) -> None:
        self.data = data
        self.sampling = sampling

    @property
    def _first_index_value(self) -> Tuple:
        if self.data is None or len(self.data) == 0:
            return None

        return next(iter(self.data))

    @property
    def _first_index_features(self) -> List[NumpyFeature]:
        if self._first_index_value is None:
            return []
        return self.data[self._first_index_value]

    @property
    def feature_count(self) -> int:
        return len(self._first_index_features)

    # TODO: Turn into function. Let's only use property for inexpensive code.
    @property
    def feature_names(self) -> List[str]:
        # Only look at the feature in the first index
        # to get the feature names. All features in all
        # indexes should have the same names
        return [feature.name for feature in self._first_index_features]

    def schema(self) -> Event:
        return Event(
            features=[
                feature.schema() for feature in list(self.data.values())[0]
            ],
            sampling=Sampling(
                index={
                    # TODO: create .schema() NumpySampling method
                    index_name: dtype.STRING
                    if index_dtype is np.str_
                    else DTYPE_MAPPING[index_dtype]
                    for index_name, index_dtype in self.sampling.index.items()
                },
                is_unix_timestamp=self.sampling.is_unix_timestamp,
            ),
        )

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        index_names: List[str] = None,
        timestamp_column: str = "timestamp",
    ) -> "NumpyEvent":
        """Convert a pandas DataFrame to a NumpyEvent.
        Args:
            df: DataFrame to convert to NumpyEvent.
            index_names: names of the DataFrame columns to be used as index for
                the event. Defaults to [].
            timestamp_column: Column containing timestamps. Supported date types:
                {np.datetime64, pd.Timestamp, datetime.datetime}. Timestamps of
                these types are converted implicitly to UTC epoch float.

        Returns:
            NumpyEvent: NumpyEvent created from DataFrame.

        Raises:
            ValueError: If index_names or timestamp_column are not in df columns.
            ValueError: If a column has an unsupported dtype.

        Example:
            >>> import pandas as pd
            >>> from temporian.implementation.numpy.data.event import NumpyEvent
            >>> df = pd.DataFrame(
            ...     data=[
            ...         [666964, 1.0, 740.0],
            ...         [666964, 2.0, 508.0],
            ...         [574016, 3.0, 573.0],
            ...     ],
            ...     columns=["product_id", "timestamp", "costs"],
            ... )
            >>> event = NumpyEvent.from_dataframe(df, index_names=["product_id"])
        """
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
        is_unix_timestamp = pd.api.types.is_datetime64_any_dtype(
            df[timestamp_column]
        )

        # convert timestamp column to float
        df[timestamp_column] = df[timestamp_column].apply(
            convert_date_to_duration
        )

        # check column dtypes, every dtype should be a key of DTYPE_MAPPING
        for column in df.columns:
            # if dtype is object, check if it only contains string values
            if df[column].dtype.type is np.object_:
                if not df[column].apply(lambda x: isinstance(x, str)).all():
                    raise ValueError(
                        "Object columns are only allowed if they contain only"
                        " string values"
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
        feature_columns = [
            column
            for column in df.columns
            if column not in index_names + [timestamp_column]
        ]

        sampling = {}
        data = {}

        # fill missing values with np.nan
        df = df.fillna(np.nan)

        # The user provided an index
        if index_names:
            group_by_indexes = df.groupby(index_names)

            for group in group_by_indexes.groups:
                columns = group_by_indexes.get_group(group)
                timestamp = columns[timestamp_column].to_numpy()

                # Convert group to tuple, useful when its only one value
                if not isinstance(group, tuple):
                    group = (group,)

                sampling[group] = timestamp

                data[group] = [
                    NumpyFeature(
                        feature,
                        columns[feature].to_numpy(
                            dtype=columns[feature].dtype.type
                        ),
                    )
                    for feature in feature_columns
                ]

        # The user did not provide an index
        else:
            timestamp = df[timestamp_column].to_numpy()
            sampling[()] = timestamp
            data[()] = [
                NumpyFeature(feature, df[feature].to_numpy())
                for feature in feature_columns
            ]

        numpy_sampling = NumpySampling(
            index={
                index_name: df.dtypes[index_name].type
                if df.dtypes[index_name].type is not str
                else np.str_
                for index_name in index_names
            },
            data=sampling,
            is_unix_timestamp=is_unix_timestamp,
        )

        return NumpyEvent(data=data, sampling=numpy_sampling)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert a NumpyEvent to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame created from NumpyEvent.
        """
        # Creating an empty dictionary to store the data
        data = {}

        columns = list(self.sampling.index) + self.feature_names + ["timestamp"]
        for column_name in columns:
            data[column_name] = []

        for index, features in self.data.items():
            timestamps = self.sampling.data[index]
            data["timestamp"].extend(timestamps)
            for feature in features:
                data[feature.name].extend(feature.data)

            if not isinstance(index, tuple):
                index = (index,)

            for i, index_key in enumerate(self.sampling.index):
                data[index_key].extend([index[i]] * len(timestamps))

        # Converting dictionary to pandas DataFrame
        df = pd.DataFrame(data)

        return df

    def __repr__(self) -> str:
        def repr_features(features: list[NumpyFeature]) -> str:
            """Repr for a list of features."""

            feature_repr = []
            for idx, f in enumerate(features):
                if idx > MAX_NUM_PRINTED_FEATURES:
                    feature_repr.append("...")
                    break
                feature_repr.append(f"{f.name} <{f.dtype}>: {f.data}")
            return "\n".join(feature_repr)

        # Representation of the "data" field
        with np.printoptions(precision=4, threshold=6):
            data_repr = []
            for idx, (k, v) in enumerate(self.data.items()):
                if idx > MAX_NUM_PRINTED_INDEX:
                    data_repr.append("...")
                    break
                data_repr.append(f"{k}:\n{string.indent(repr_features(v))}")
            data_repr = string.indent("\n".join(data_repr))

        # Representation of the "sampling" field
        sampling_repr = string.indent(self.sampling.__repr__())

        return (
            f"data ({len(self.data)}):\n{data_repr}\nsampling:\n{sampling_repr}"
        )

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, NumpyEvent):
            return False

        # Check equal sampling and index values
        if self.sampling != __o.sampling:
            return False

        # Check same features
        if self.feature_names != __o.feature_names:
            return False

        # Check each feature is equal in each index
        for index in self.data.keys():
            # Check both feature list are equal
            if self.data[index] != __o.data[index]:
                return False

        return True

    def index(self) -> Sequence[Any]:
        """Sequence of available indexes."""

        return self.data.keys()

    def plot(self, *args, **wargs) -> Any:
        """Plots an event. See tp.plot for details."""

        from temporian.implementation.numpy.data import plotter

        return plotter.plot(event=self, *args, **wargs)
