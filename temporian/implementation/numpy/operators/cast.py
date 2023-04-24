from typing import Dict

import numpy as np

from temporian.core.data.dtype import DType
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
        self._dtype_limits = {
            DType.INT32: np.iinfo(np.int32),
            DType.INT64: np.iinfo(np.int64),
            DType.FLOAT32: np.finfo(np.float32),
            DType.FLOAT64: np.finfo(np.float64),
        }

        self._nocheck_dtypes = [DType.BOOLEAN, DType.STRING]

    def _can_overflow(self, origin_dtype: DType, dst_dtype: DType) -> bool:
        # Don't check overflow for BOOLEAN or STRING:
        #  - boolean: makes no sense, everybody knows what to expect.
        #  - string: on orig_dtype, too costly to convert to numeric dtype
        #            and compare to the limit. On dst_type, there's no limit.
        if (
            origin_dtype in self._nocheck_dtypes
            or dst_dtype in self._nocheck_dtypes
        ):
            return False
        return (
            self._dtype_limits[origin_dtype].max
            > self._dtype_limits[dst_dtype].max
        )

    def _check_overflow(
        self,
        data: np.ndarray,
        origin_dtype: DType,
        dst_dtype: DType,
    ) -> None:
        if np.any(
            (data < self._dtype_limits[dst_dtype].min)
            | (data > self._dtype_limits[dst_dtype].max)
        ):
            raise ValueError(
                f"Overflow casting {origin_dtype}->{dst_dtype} {data=}"
            )

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        from_features = self.operator.attributes["from_features"]
        check = self.operator.attributes["check_overflow"]

        # Reuse event if actually no features changed dtype
        operator: CastOperator = self.operator
        if operator.reuse_event:
            return {"event": event}

        # Create new event, some features may be reused
        # NOTE: it's currently faster in the benchmark to run feat/event_idx,
        # but this might need a re-check with future implementations.
        output = NumpyEvent(data={}, sampling=event.sampling)
        for feat_idx, feature_name in enumerate(event.feature_names()):
            dst_dtype = DType(from_features[feature_name])
            orig_dtype = event.dtypes[feature_name]
            check_feature = check and self._can_overflow(orig_dtype, dst_dtype)
            # Numpy dest type
            dst_dtype_np = DTYPE_REVERSE_MAPPING[dst_dtype]
            for event_index, features in event.data.items():
                feature = features[feat_idx]
                # Initialize row with first feature
                if feat_idx == 0:
                    idx_features = []
                    output.data[event_index] = idx_features
                else:
                    idx_features = output.data[event_index]

                # Reuse if both features have the same dtype
                if feature.dtype == dst_dtype:
                    idx_features.append(feature)
                else:
                    data = feature.data
                    # Check overflow when needed
                    if check_feature:
                        self._check_overflow(data, orig_dtype, dst_dtype)

                    # Create new feature
                    idx_features.append(
                        NumpyFeature(
                            name=feature_name,
                            data=data.astype(dst_dtype_np),
                        )
                    )

        return {"event": output}


implementation_lib.register_operator_implementation(
    CastOperator, CastNumpyImplementation
)
