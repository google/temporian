# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import numpy as np

from temporian.core.operators.sample import Sample
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event import DTYPE_REVERSE_MAPPING
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy_cc.operators import sample as sample_cc
from temporian.implementation.numpy.operators.base import OperatorImplementation


class SampleNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the sample operator."""

    def __init__(self, operator: Sample) -> None:
        super().__init__(operator)
        assert isinstance(operator, Sample)

    def __call__(
        self, event: NumpyEvent, sampling: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        # Type and replacement values
        output_features = self._operator.outputs["event"].features
        output_missing_and_np_dtypes = [
            (
                f.dtype.missing_value(),
                DTYPE_REVERSE_MAPPING[f.dtype],
            )
            for f in output_features
        ]
        # create output event
        dst_event = NumpyEvent(
            data={},
            feature_names=event.feature_names,
            index_names=event.index_names,
            is_unix_timestamp=event.is_unix_timestamp,
        )
        # iterate over destination sampling
        for index_key, index_data in sampling.iterindex():
            # intialize destination index data
            dst_mts = []
            dst_event[index_key] = IndexData(dst_mts, index_data.timestamps)

            if index_key not in event.data:
                # No matching events to sample from
                for (
                    output_missing_value,
                    output_np_dtype,
                ) in output_missing_and_np_dtypes:
                    dst_mts.append(
                        np.full(
                            shape=len(index_data),
                            fill_value=output_missing_value,
                            dtype=output_np_dtype,
                        )
                    )
                continue

            src_mts = event[index_key].features
            src_timestamps = event[index_key].timestamps
            (
                sampling_idxs,
                first_valid_idx,
            ) = sample_cc.build_sampling_idxs(
                src_timestamps, index_data.timestamps
            )
            # For each feature
            for src_ts, (output_missing_value, output_np_dtype) in zip(
                src_mts, output_missing_and_np_dtypes
            ):
                # TODO: Check if running the following block in c++ is faster.
                dst_ts_data = np.full(
                    shape=len(index_data),
                    fill_value=output_missing_value,
                    dtype=src_ts.dtype,
                )
                dst_ts_data[first_valid_idx:] = src_ts[
                    sampling_idxs[first_valid_idx:]
                ]
                dst_mts.append(dst_ts_data)

        return {"event": dst_event}


implementation_lib.register_operator_implementation(
    Sample, SampleNumpyImplementation
)
