"""Implementation for the Sample operator."""


from typing import Dict
import numpy as np

from temporian.implementation.numpy.data.event import (
    NumpyEvent,
    NumpyFeature,
)
from temporian.core.operators.prefix import Prefix
from temporian.implementation.numpy import implementation_lib


class PrefixNumpyImplementation:
    """Numpy implementation for the Prefix operator."""

    def __init__(self, operator: Prefix) -> None:
        assert isinstance(operator, Prefix)
        self._operator = operator

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        prefix = self._operator.prefix()
        dst_event = NumpyEvent(data={}, sampling=event.sampling)

        # For each index value
        for index, src_mts in event.data.items():
            dst_mts = []
            dst_event.data[index] = dst_mts

            # For each feature
            for src_ts in src_mts:
                dst_mts.append(NumpyFeature(prefix + src_ts.name, src_ts.data))

        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    Prefix, PrefixNumpyImplementation
)
