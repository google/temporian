from typing import Dict, List, Tuple

import numpy as np


class NumpySampling:
    def __init__(self, names: List[str], data: Dict[Tuple, np.ndarray]) -> None:
        self.names = names
        self.data = data
