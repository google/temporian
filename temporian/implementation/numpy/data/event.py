from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data import dtype
from temporian.implementation.numpy.data.sampling import NumpySampling

DTYPE_MAPPING = {
    np.float64: dtype.FLOAT64,
    np.float32: dtype.FLOAT32,
    np.int64: dtype.INT64,
    np.int32: dtype.INT32,
}


class NumpyFeature:
    def __init__(self, name: str, data: np.ndarray) -> None:
        if len(data.shape) > 1:
            raise ValueError(
                "NumpyFeatures can only be created from flat arrays. Passed"
                f" input's shape: {len(data.shape)}"
            )
        if data.dtype.type is not np.string_:
            if data.dtype.type not in DTYPE_MAPPING:
                raise ValueError(
                    f"Unsupported dtype {data.dtype} for NumpyFeature."
                    f" Supported dtypes: {DTYPE_MAPPING.keys()}"
                )

        self.name = name
        self.data = data
        self.dtype = data.dtype.type

    def __repr__(self) -> str:
        return f"{self.name}: {self.data.__repr__()}"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, NumpyFeature):
            return False

        if self.name != __o.name:
            return False

        if not np.array_equal(self.data, __o.data, equal_nan=True):
            return False

        return True

    def core_dtype(self) -> Any:
        if self.dtype.type is np.string_:
            return dtype.STRING
        return DTYPE_MAPPING[self.dtype]


class NumpyEvent:
    def __init__(
        self,
        data: Dict[Tuple, List[NumpyFeature]],
        sampling: NumpySampling,
    ) -> None:
        self.data = data
        self.sampling = sampling

    @property
    def first_index_level(self) -> Tuple:
        first_index_level = None
        try:
            first_index_level = next(iter(self.data))
        except StopIteration:
            return None

        return first_index_level

    @property
    def feature_count(self) -> int:
        if len(self.data.keys()) == 0:
            return 0

        return len(self.data[self.first_index_level])

    @property
    def feature_names(self) -> List[str]:
        if len(self.data.keys()) == 0:
            return []

        # Only look at the feature in the first index
        # to get the feature names. All features in all
        # indexes should have the same names
        return [feature.name for feature in self.data[self.first_index_level]]

    def schema(self) -> Event:
        return Event(
            features=[
                feature.schema() for feature in list(self.data.values())[0]
            ],
            sampling=self.sampling.names,
        )

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        index_names: List[str],
        timestamp_name: str = "timestamp",
    ) -> "NumpyEvent":
        """Function to convert a pandas DataFrame to a NumpyEvent

        Args:
            df: DataFrame to convert to NumpyEvent
            index_names: Names of the indexes of the DataFrame
            timestamp_name: Name for timestamp index. Defaults to "timestamp".

        Returns:
            NumpyEvent: NumpyEvent created from DataFrame
        """
        # check index names and timestamp name are in df columns
        if (
            not all(index_name in df.columns for index_name in index_names)
            or timestamp_name not in df.columns
        ):
            raise ValueError(
                f"Index names {index_names} and timestamp name {timestamp_name}"
                " are not in DataFrame columns"
            )

        feature_columns = [
            column
            for column in df.columns
            if column not in index_names + [timestamp_name]
        ]

        sampling = {}
        data = {}

        if index_names:
            group_by_indexes = df.groupby(index_names)

            for group in group_by_indexes.groups:
                columns = group_by_indexes.get_group(group)
                timestamp = columns[timestamp_name].to_numpy()

                # Convert group to tuple, useful when its only one value
                if not isinstance(group, tuple):
                    group = (group,)

                sampling[group] = timestamp
                data[group] = [
                    NumpyFeature(feature, columns[feature].to_numpy())
                    for feature in feature_columns
                ]
        else:
            timestamp = df[timestamp_name].to_numpy()
            sampling[()] = timestamp
            data[()] = [
                NumpyFeature(feature, df[feature].to_numpy())
                for feature in feature_columns
            ]

        numpy_sampling = NumpySampling(names=index_names, data=sampling)

        return NumpyEvent(data=data, sampling=numpy_sampling)

    def to_dataframe(
        self, timestamp_index_name: str = "timestamp"
    ) -> pd.DataFrame:
        """Function to convert a NumpyEvent to a pandas DataFrame

        Args:
            timestamp_index_name: Name for timestamp index. Defaults to "timestamp".

        Returns:
            pd.DataFrame: DataFrame created from NumpyEvent
        """
        df_index = self.sampling.names + [timestamp_index_name]
        df_features = self.feature_names
        columns = df_index + df_features

        df = pd.DataFrame(data=[], columns=columns).set_index(df_index)

        for index, features in self.data.items():
            timestamps = self.sampling.data[index]

            for i, timestamp in enumerate(timestamps):
                # If no index, index is timestamp
                if len(index) == 0:
                    new_index = timestamp
                # add timestamp to index
                else:
                    new_index = index + (timestamp,)

                df.loc[new_index, df_features] = [
                    feature.data[i] for feature in features
                ]

        # Convert to original dtypes, can be more efficient
        first_index = self.first_index_level
        first_features = self.data[first_index]
        df = df.astype(
            {feature.name: feature.data[0].dtype for feature in first_features}
        )

        return df

    def __repr__(self) -> str:
        return self.data.__repr__() + " " + self.sampling.__repr__()

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
