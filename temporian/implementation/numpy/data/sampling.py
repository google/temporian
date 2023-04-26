from typing import Dict, List, Tuple

import numpy as np

from temporian.utils import string

# Maximum of printed index when calling repr(event)
MAX_NUM_PRINTED_INDEX = 5


class NumpySampling:
    def __init__(
        self,
        index: List[str],
        data: Dict[Tuple, np.ndarray],
        is_unix_timestamp: bool = False,
    ) -> None:
        self.index = index
        self.data = data
        self.is_unix_timestamp = is_unix_timestamp

    @property
    def has_repeated_timestamps(self) -> bool:
        """Check if any index has repeated timestamps.

        Returns:
            `True` if any index has repeated timestamps.
        """
        for index in self.data:
            if len(self.data[index]) != len(np.unique(self.data[index])):
                return True

        return False

    def __repr__(self) -> str:
        with np.printoptions(precision=4, threshold=6):
            data_repr = []
            for idx, (k, v) in enumerate(self.data.items()):
                if idx > MAX_NUM_PRINTED_INDEX:
                    data_repr.append("...")
                    break
                data_repr.append(f"{k}: {v}")
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
