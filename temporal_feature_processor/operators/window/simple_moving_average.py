from typing import Union

from pandas import Timedelta

from temporal_feature_processor.sampling import Sampling
from temporal_feature_processor.sequences import EventSequence, FeatureSequence

from .base import WindowOperator


class SimpleMovingAverageOperator(WindowOperator):
    """Base class for window operators."""

    def __init__(self, sampling: Sampling, window_length: Timedelta) -> None:
        super().__init__(sampling=sampling, window_length=window_length)

    def __call__(
        self,
        input: Union[EventSequence, FeatureSequence],
    ) -> Union[EventSequence, FeatureSequence]:
        """Apply a simple moving average to an event or feature.
        If input is an event, the moving average will be computed for each of its features independently.

        Args:
            input (Union[EventSequence, FeatureSequence]): the input sequence to apply a simple moving average to.

        Returns:
            Union[EventSequence, FeatureSequence]: the output of the operator.
                The output sequence will have the same type as the input sequence.
        """
        # TODO: implement logic
        pass
