"""Implementation for the Sample operator."""

from typing import Dict

import numpy as np

from temporian.core.operators.sample import Sample
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.feature import dtype_to_np_dtype
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy_cc.operators import sample as sample_cc
from temporian.implementation.numpy.operators.base import OperatorImplementation


class SampleNumpyImplementation(OperatorImplementation):
    """Numpy implementation for the Sample operator."""

    def __init__(self, operator: Sample) -> None:
        assert isinstance(operator, Sample)
        super().__init__(operator)

    def __call__(
        self, event: NumpyEvent, sampling: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        # Type and replacement values
        output_features = self._operator.outputs["event"].features
        output_missing_and_np_dtypes = [
            (
                f.dtype.missing_value(),
                dtype_to_np_dtype(f.dtype),
            )
            for f in output_features
        ]
        dst_event = NumpyEvent(
            data={},
            feature_names=event.feature_names,
            index_names=event.index_names,
        )
        for index_key, index_data in sampling.iterindex():
            dst_mts = []
            dst_event.data[index_key] = IndexData(
                dst_mts, index_data.timestamps
            )
            sampling_timestamps = index_data.timestamps

            if index_key not in event.data:
                # No matchin events to sample from.
                for (
                    output_missing_value,
                    output_np_dtype,
                ) in output_missing_and_np_dtypes:
                    dst_ts_data = np.full(
                        shape=len(sampling_timestamps),
                        fill_value=output_missing_value,
                        dtype=output_np_dtype,
                    )
                    dst_mts.append(dst_ts_data)
                continue

            src_mts = event.data[index_key].features
            src_timestamps = event.data[index_key].timestamps
            (
                sampling_idxs,
                first_valid_idx,
            ) = sample_cc.build_sampling_idxs(
                src_timestamps, sampling_timestamps
            )
            # For each feature
            for src_ts, (output_missing_value, output_np_dtype) in zip(
                src_mts, output_missing_and_np_dtypes
            ):
                # TODO: Check if running the following block in c++ is faster.
                dst_ts_data = np.full(
                    shape=len(sampling_timestamps),
                    fill_value=output_missing_value,
                    dtype=src_ts.data.dtype,
                )
                dst_ts_data[first_valid_idx:] = src_ts[
                    sampling_idxs[first_valid_idx:]
                ]
                dst_mts.append(dst_ts_data)

        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    Sample, SampleNumpyImplementation
)
