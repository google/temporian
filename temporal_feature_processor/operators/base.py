from abc import ABC, abstractmethod
from typing import Any

from temporal_feature_processor.sequences import EventSequence


class Operator(ABC):
    """Base class to define an operator's interface."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> EventSequence:
        """Apply the operator to its inputs.

        Returns:
            EventSequence: the output event of the operator.
        """
