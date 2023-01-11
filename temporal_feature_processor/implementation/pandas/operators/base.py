from abc import ABC, abstractmethod
from typing import Any

from temporal_feature_processor.implementation.pandas.data.event import PandasEvent


class PandasOperator(ABC):
    """Base class to define an operator's interface."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> PandasEvent:
        """Apply the operator to its inputs.

        Returns:
            PandasEvent: the output event of the operator.
        """
