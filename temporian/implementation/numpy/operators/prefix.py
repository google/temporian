"""Implementation for the Sample operator."""


from typing import Dict

from temporian.core.operators.prefix import Prefix
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.base import OperatorImplementation


class PrefixNumpyImplementation(OperatorImplementation):
    """Numpy implementation for the Prefix operator."""

    def __init__(self, operator: Prefix) -> None:
        super().__init__(operator)
        assert isinstance(operator, Prefix)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        # gather operator attributes
        prefix = self._operator.prefix()

        # create output event
        dst_event = NumpyEvent(
            data=event.data,
            feature_names=[
                f"{prefix}{feature_name}"
                for feature_name in event.feature_names
            ],
            index_names=event.index_names,
            is_unix_timestamp=event.is_unix_timestamp,
        )
        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    Prefix, PrefixNumpyImplementation
)
