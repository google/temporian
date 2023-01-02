from pandas import Timestamp

from temporal_feature_processor.interval import Interval

from .base import Sampling


class UniformSampling(Sampling):
    def __init__(self, interval: Interval, start: Timestamp, end: Timestamp) -> None:
        self.interval = interval
        self.start = start
        self.end = end
