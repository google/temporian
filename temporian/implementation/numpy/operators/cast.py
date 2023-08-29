from typing import Dict, Tuple, List, Optional, Any

import numpy as np

from temporian.core.data.dtype import DType
from temporian.core.operators.cast import CastOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.dtype_normalization import (
    tp_dtype_to_np_dtype,
)
from temporian.implementation.numpy.data.event_set import EventSet, IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation

_DTYPE_LIMITS = {
    DType.INT32: np.iinfo(np.int32),
    DType.INT64: np.iinfo(np.int64),
    DType.FLOAT32: np.finfo(np.float32),
    DType.FLOAT64: np.finfo(np.float64),
}

_NO_CHECK_TYPES = [DType.BOOLEAN, DType.STRING]


def _can_overflow(origin_dtype: DType, dst_dtype: DType) -> bool:
    """Tests if a cast should be tested for overflow.

    Don't check overflow for BOOLEAN or STRING:
      - boolean: makes no sense, everybody knows what to expect.
      - string: on src_dtype, too costly to convert to numeric dtype
                and compare to the limit. On dst_type, there's no limit.
    """
    if origin_dtype in _NO_CHECK_TYPES or dst_dtype in _NO_CHECK_TYPES:
        return False
    return _DTYPE_LIMITS[origin_dtype].max > _DTYPE_LIMITS[dst_dtype].max


def _check_overflow(
    data: np.ndarray,
    origin_dtype: DType,
    dst_dtype: DType,
    feature_name: str,
    min_max: Tuple,
):
    if np.any((data < min_max[0]) | (data > min_max[1])):
        raise ValueError(
            f"Overflow when casting feature {feature_name!r} from dtype"
            f" {origin_dtype} to dtype {dst_dtype} {data=}. You can disable"
            " overflow checking with check_overflow=False."
        )


class CastNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: CastOperator) -> None:
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, CastOperator)
        output_schema = self.output_schema("output")

        # Reuse evset if actually no features changed dtype
        if self.operator.is_noop:
            return {"output": input}

        # Min/max ranges for each of the features. If None, no check is done.
        mins_maxs: List[Optional[Tuple[Any, Any]]] = []
        for src_feature, dst_dtype in zip(
            input.schema.features, self.operator.dtypes
        ):
            if self.operator.check_overflow and _can_overflow(
                src_feature.dtype, dst_dtype
            ):
                iinfo = _DTYPE_LIMITS[dst_dtype]
                mins_maxs.append((iinfo.min, iinfo.max))
            else:
                mins_maxs.append(None)

        # Numpy output dtype for each feature.
        np_dtypes = [
            tp_dtype_to_np_dtype(tp_dtype) for tp_dtype in self.operator.dtypes
        ]

        output_evset = EventSet(data={}, schema=output_schema)
        for index_key, index_data in input.data.items():
            dst_features = []
            for feature_idx, (
                min_max,
                np_dtype,
                src_schema,
                dst_schema,
            ) in enumerate(
                zip(
                    mins_maxs,
                    np_dtypes,
                    input.schema.features,
                    output_schema.features,
                )
            ):
                src_values = index_data.features[feature_idx]
                if min_max is not None:
                    _check_overflow(
                        src_values,
                        src_schema.dtype,
                        dst_schema.dtype,
                        src_schema.name,
                        min_max,
                    )
                dst_features.append(src_values.astype(np_dtype))

            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=dst_features,
                    timestamps=index_data.timestamps,
                    schema=output_schema,
                ),
                normalize=False,
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    CastOperator, CastNumpyImplementation
)
