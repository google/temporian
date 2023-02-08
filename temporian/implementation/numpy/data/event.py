from typing import Dict, List, Tuple, Any

import numpy as np

from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data import dtype
from temporian.implementation.numpy.data.sampling import NumpySampling


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
            name=self.name, dtype=self.dtype, sampling=self.sampling.names
        )

    def __repr__(self) -> str:
        return f"{self.name}: {self.data.__repr__()}"

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

    def schema(self) -> Event:
        return Event(
            features=[
                feature.schema() for feature in list(self.data.values())[0]
            ],
            sampling=self.sampling.names,
        )

    def __repr__(self) -> str:
        return self.data.__repr__()


DTYPE_MAPPING = {
    np.float64: dtype.FLOAT64,
    np.float32: dtype.FLOAT32,
    np.int64: dtype.INT64,
    np.int32: dtype.INT32,
}
