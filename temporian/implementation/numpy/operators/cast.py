from typing import Dict

import numpy as np

from temporian.core.data.dtype import DType
from temporian.core.operators.cast import CastOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import DTYPE_MAPPING
from temporian.implementation.numpy.data.event_set import DTYPE_REVERSE_MAPPING
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
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
        #  - string: on src_dtype, too costly to convert to numeric dtype
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

    def __call__(self, node: EventSet) -> Dict[str, EventSet]:
        from_features = self.operator.attributes["from_features"]
        check = self.operator.attributes["check_overflow"]

        # Reuse evset if actually no features changed dtype
        operator: CastOperator = self.operator
        if operator.reuse_node:
            return {"node": node}

        # Create new evset, some features may be reused
        # NOTE: it's currently faster in the benchmark to run feat/event_idx,
        # but this might need a re-check with future implementations.
        dst_evset = EventSet(
            data={},
            feature_names=node.feature_names,
            index_names=node.index_names,
            is_unix_timestamp=node.is_unix_timestamp,
        )
        for feat_idx, feature_name in enumerate(node.feature_names):
            src_dtype = node.dtypes[feature_name]
            dst_dtype = DType(from_features[feature_name])
            check_feature = check and self._can_overflow(src_dtype, dst_dtype)
            # Numpy destination type
            dst_dtype_np = DTYPE_REVERSE_MAPPING[dst_dtype]
            for index_key, index_data in node.iterindex():
                feature = index_data.features[feat_idx]
                # Initialize row with first feature
                if feat_idx == 0:
                    idx_features = []
                    dst_evset[index_key] = IndexData(
                        idx_features, index_data.timestamps
                    )
                else:
                    idx_features = dst_evset[index_key].features

                # Reuse if both features have the same dtype
                if DTYPE_MAPPING[feature.dtype.type] == dst_dtype:
                    idx_features.append(feature)
                else:
                    # Check overflow when needed
                    if check_feature:
                        self._check_overflow(feature, src_dtype, dst_dtype)

                    # Create new feature
                    idx_features.append(
                        feature.astype(dst_dtype_np),
                    )

        return {"node": dst_evset}


implementation_lib.register_operator_implementation(
    CastOperator, CastNumpyImplementation
)
