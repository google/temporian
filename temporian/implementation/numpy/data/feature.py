from typing import Any

import numpy as np
from temporian.core.data.event import Feature
from temporian.core.data import dtype

DTYPE_MAPPING = {
    np.float64: dtype.FLOAT64,
    np.float32: dtype.FLOAT32,
    np.int64: dtype.INT64,
    np.int32: dtype.INT32,
    np.bool_: dtype.BOOLEAN,
}

DTYPE_REVERSE_MAPPING = {v: k for k, v in DTYPE_MAPPING.items()}
DTYPE_REVERSE_MAPPING[dtype.STRING] = np.str_


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

        self._name = name
        self._data = data

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray):
        self._data = new_data

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
