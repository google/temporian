from pandas import Timedelta

from temporal_feature_processor.sampling import Sampling
from temporal_feature_processor.sequences import EventSequence

from .base import WindowOperator


class SimpleMovingAverageOperator(WindowOperator):
    """Base class for window operators."""

    def __init__(self, sampling: Sampling, window_length: Timedelta) -> None:
        super().__init__(sampling=sampling, window_length=window_length)

    def __call__(
        self,
        input: EventSequence,
    ) -> EventSequence:
        """Apply a simple moving average to an event.
        If input has more than one feature, the moving average will be computed for each of its features independently.

        Args:
            input (EventSequence): the input event to apply a simple moving average to.

        Returns:
            EventSequence: the output of the operator.
        """
        # TODO: implement logic
        pass
