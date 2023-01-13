from temporal_feature_processor.implementation.pandas.data.event import PandasEvent
from temporal_feature_processor.implementation.pandas.sampling import PandasSampling

from .base import PandasWindowOperator


class PandasSimpleMovingAverageOperator(PandasWindowOperator):
  """Base class for window operators."""

  def __init__(self, window_length: int) -> None:
    super().__init__(window_length=window_length)

  def __call__(
      self,
      input: PandasEvent,
      sampling: PandasSampling,
  ) -> PandasEvent:
    """Apply a simple moving average to an event.
        If input has more than one feature, the moving average will be computed for each of its features independently.

        Args:
            input (PandasEvent): the input event to apply a simple moving average to.

        Returns:
            PandasEvent: the output of the operator.
        """
    # TODO: implement logic
    pass
