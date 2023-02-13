from typing import Dict, List, Tuple, Any

import numpy as np

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
    def __init__(
        self, name: str, sampling: NumpySampling, data: np.ndarray
    ) -> None:
        if len(data.shape) > 1:
            raise ValueError(
                "NumpyFeatures can only be created from flat arrays. Passed"
                f" input's shape: {len(data.shape)}"
            )
        if data.dtype.type not in DTYPE_MAPPING:
            raise ValueError(
                f"Unsupported dtype {data.dtype} for NumpyFeature. Supported"
                f" dtypes: {DTYPE_MAPPING.keys()}"
            )

        self.name = name
        self.sampling = sampling
        self.data = data
        self.dtype = data.dtype.type

    def schema(self) -> Feature:
        return Feature(
            name=self.name,
            dtype=self.core_dtype(),
            sampling=self.sampling.names,
        )

    def __repr__(self) -> str:
        return f"{self.name}: {self.data.__repr__()}"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, NumpyFeature):
            return False

        if self.name != __o.name:
            return False

        if self.sampling != __o.sampling:
            return False

        if not np.array_equal(self.data, __o.data, equal_nan=True):
            return False

        return True

    def core_dtype(self) -> Any:
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
    def feature_count(self) -> int:
        if len(self.data.keys()) == 0:
            return 0

        first_index = next(iter(self.data))
        return len(self.data[first_index])

    @property
    def feature_names(self) -> List[str]:
        if len(self.data.keys()) == 0:
            return []

        # Only look at the feature in the first index
        # to get the feature names. All features in all
        # indexes should have the same names
        first_index = next(iter(self.data))
        return [feature.name for feature in self.data[first_index]]

    def schema(self) -> Event:
        return Event(
            features=[
                feature.schema() for feature in list(self.data.values())[0]
            ],
            sampling=self.sampling.names,
        )

    def __repr__(self) -> str:
        return self.data.__repr__()

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
