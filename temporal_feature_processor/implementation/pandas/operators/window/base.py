from temporal_feature_processor.implementation.pandas.operators.base import (
    PandasOperator,)


class PandasWindowOperator(PandasOperator):
  """Base class for pandas window operator implementations."""

  def __init__(self, window_length: int) -> None:
    super().__init__()
    self.window_length = window_length
