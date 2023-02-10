from typing import Dict, List, Tuple

import numpy as np


class NumpySampling:
    def __init__(self, names: List[str], data: Dict[Tuple, np.ndarray]) -> None:
        self.names = names
        self.data = data

    def __eq__(self, other):

        if not isinstance(other, NumpySampling):
            return False

        if self.names != other.names:
            return False

        if self.data.keys() != other.data.keys():
            return False

         # Check if both sampling have same timestamps per index
        for index in self.data:
            if not np.array_equal(self.data[index], other.data[index]):
                return False

        return True
