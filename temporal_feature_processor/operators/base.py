from abc import ABC, abstractmethod
from typing import Any, Union

from temporal_feature_processor.sequences import EventSequence, FeatureSequence


class Operator(ABC):
    """Base class to define an operator's interface."""

    @abstractmethod
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Union[EventSequence, FeatureSequence]:
        """Apply the operator to its inputs.

        Returns:
            Union[EventSequence, FeatureSequence]: the output of the operator.
        """
