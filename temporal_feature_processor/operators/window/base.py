from pandas import Timedelta

from temporal_feature_processor.sampling import Sampling

from ..base import Operator


class WindowOperator(Operator):
    """Base class for window operators."""

    def __init__(self, sampling: Sampling, window_length: Timedelta) -> None:
        super().__init__()
        self.sampling = sampling
        self.window_length = window_length
