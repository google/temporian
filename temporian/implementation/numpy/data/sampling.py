from typing import Dict, List, Tuple

import numpy as np


class NumpySampling:
    def __init__(self, index: List[str], data: Dict[Tuple, np.ndarray]) -> None:
        self.index = index
        self.data = data

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

    def __repr__(self) -> str:
        return f"index:{self.index} data:{self.data.__repr__()}"

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
