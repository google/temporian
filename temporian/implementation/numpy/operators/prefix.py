"""Implementation for the Sample operator."""


from typing import Dict

from temporian.implementation.numpy.data.event import (
    NumpyEvent,
    NumpyFeature,
)
from temporian.core.operators.prefix import Prefix
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class PrefixNumpyImplementation(OperatorImplementation):
    """Numpy implementation for the Prefix operator."""

    def __init__(self, operator: Prefix) -> None:
        assert isinstance(operator, Prefix)
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        prefix = self._operator.prefix()
        dst_event = NumpyEvent(data={}, sampling=event.sampling)

        # For each index value
        for index, features in event.data.items():
            dst_event.data[index] = [
                NumpyFeature(prefix + feature.name, feature.data)
                for feature in features
            ]

        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    Prefix, PrefixNumpyImplementation
)
