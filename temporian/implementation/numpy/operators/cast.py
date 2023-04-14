import numpy as np
import temporian.core.data.dtype as tp_dtypes

from typing import Dict
from temporian.core.operators.cast import CastOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.feature import (
    NumpyFeature,
    DTYPE_REVERSE_MAPPING,
)
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class CastNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: CastOperator) -> None:
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        target_dtypes = self.operator.attributes["target_dtypes"]
        check_overflow = self.operator.attributes["check_overflow"]
        dtype_limits = {
            tp_dtypes.INT32: np.iinfo(np.int32),
            tp_dtypes.FLOAT32: np.finfo(np.float32),
            tp_dtypes.INT64: np.iinfo(np.int64),  # may overflow from float64
        }

        # Reuse event if actually no features changed dtype
        if all(
            orig_dtype is target_dtypes[feature_name]
            for feature_name, orig_dtype in event.dtypes.items()
        ):
            return {"event": event}

        # Create new event, some features may be reused
        output = NumpyEvent(data={}, sampling=event.sampling)

        for event_index, features in event.data.items():
            output.data[event_index] = []

            for feature in features:
                # Reuse if both features have the same dtype
                tp_dtype = target_dtypes[feature.name]
                if feature.dtype == tp_dtype:
                    output.data[event_index].append(feature)
                else:
                    # Check overflow when needed
                    if check_overflow and tp_dtype in dtype_limits:
                        if np.any(
                            (feature.data < dtype_limits[tp_dtype].min)
                            | (feature.data > dtype_limits[tp_dtype].max)
                        ):
                            raise ValueError(
                                f"Overflow casting to {tp_dtype} at index"
                                f" {event_index}: {feature.data}"
                            )
                    # Create new feature
                    output.data[event_index].append(
                        NumpyFeature(
                            name=feature.name,  # Note: not renaming feature
                            data=feature.data.astype(
                                DTYPE_REVERSE_MAPPING[tp_dtype]
                            ),
                        )
                    )

        return {"event": output}


implementation_lib.register_operator_implementation(
    CastOperator, CastNumpyImplementation
)
