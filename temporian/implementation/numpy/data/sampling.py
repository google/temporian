from typing import Dict, List, Tuple

import numpy as np

from temporian.core.data.dtype import DType
from temporian.utils import string

# Maximum of printed index when calling repr(event)
MAX_NUM_PRINTED_INDEX = 5

# same as in numpy/data/event.py, can't import due to circular import error
PYTHON_DTYPE_MAPPING = {
    str: DType.STRING,
    # TODO: fix this, int doesn't have to be INT64 necessarily
    int: DType.INT64,
    np.int64: DType.INT64,
}


class NumpySampling:
    def __init__(
        self,
        index: List[str],
        data: Dict[Tuple, np.ndarray],
        is_unix_timestamp: bool = False,
    ) -> None:
        self._index = index
        self._data = data
        self._is_unix_timestamp = is_unix_timestamp

    @property
    def index(self) -> List[str]:
        return self._index

    @property
    def data(self) -> Dict[Tuple, np.ndarray]:
        return self._data

    @property
    def is_unix_timestamp(self) -> bool:
        return self._is_unix_timestamp

    @property
    def has_repeated_timestamps(self) -> bool:
        """Check if any index has repeated timestamps

        Returns:
            bool: True if any index has repeated timestamps
        """
        for index in self.data:
            if len(self.data[index]) != len(np.unique(self.data[index])):
                return True

        return False

    # TODO: To remove
    @property
    def dtypes(self) -> Dict[str, DType]:
        first_idx_lvl = next(iter(self.data))
        return {
            name: PYTHON_DTYPE_MAPPING[type(value)]
            for name, value in zip(self.index, first_idx_lvl)
        }

    # TODO: Rename to "dtypes".
    def dtypes_list(self) -> List[DType]:
        # TODO: Handle case where there is no data.
        # TODO: Handle non supported type PYTHON_DTYPE_MAPPING.
        first_idx_lvl = next(iter(self.data))
        return [PYTHON_DTYPE_MAPPING[type(value)] for value in first_idx_lvl]

    def __repr__(self) -> str:
        with np.printoptions(precision=4, threshold=20):
            data_repr = []
            for idx, (k, v) in enumerate(self.data.items()):
                if idx > MAX_NUM_PRINTED_INDEX:
                    data_repr.append("...")
                    break
                data_repr.append(f"{k} ({len(v)}): {v}")
            data_repr = string.indent("\n".join(data_repr))
        return f"index: {self.index}\ndata ({len(self.data)}):\n{data_repr}\n"

    def __eq__(self, other):
        if not isinstance(other, NumpySampling):
            return False

        if self.index != other.index:
            return False

        if self.data.keys() != other.data.keys():
            return False

        # Check if both sampling have same timestamps per index
        for index in self.data:
            if not np.array_equal(self.data[index], other.data[index]):
                return False

        return True
