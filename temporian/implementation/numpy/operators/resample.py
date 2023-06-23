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

from temporian.core.operators.resample import Resample
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import (
    IndexData,
    EventSet,
    tp_dtype_to_np_dtype,
)
from temporian.implementation.numpy_cc.operators import operators_cc
from temporian.implementation.numpy.operators.base import OperatorImplementation


class ResampleNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the sample operator."""

    def __init__(self, operator: Resample) -> None:
        super().__init__(operator)
        assert isinstance(operator, Resample)

    def __call__(
        self, input: EventSet, sampling: EventSet
    ) -> Dict[str, EventSet]:
        assert isinstance(self.operator, Resample)

        output_schema = self.output_schema("output")

        # Type and replacement values
        output_missing_and_np_dtypes = [
            (
                f.dtype.missing_value(),
                tp_dtype_to_np_dtype(f.dtype),
            )
            for f in output_schema.features
        ]
        # create output EventSet
        dst_evset = EventSet(data={}, schema=output_schema)
        # iterate over destination sampling
        for index_key, index_data in sampling.data.items():
            # intialize destination index data
            dst_mts = []

            index_data = IndexData(dst_mts, index_data.timestamps, schema=None)
            dst_evset[index_key] = index_data

            if index_key not in input.data:
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
                index_data.check_schema(output_schema)
                continue

            src_mts = input[index_key].features
            src_timestamps = input[index_key].timestamps
            (
                sampling_idxs,
                first_valid_idx,
            ) = operators_cc.build_sampling_idxs(
                src_timestamps, index_data.timestamps
            )
            # For each feature
            for src_ts, (output_missing_value, output_np_dtype) in zip(
                src_mts, output_missing_and_np_dtypes
            ):
                # TODO: Check if running the following block in c++ is faster.
                assert src_ts.dtype.type == output_np_dtype
                dst_ts_data = np.full(
                    shape=len(index_data),
                    fill_value=output_missing_value,
                    dtype=src_ts.dtype,
                )
                dst_ts_data[first_valid_idx:] = src_ts[
                    sampling_idxs[first_valid_idx:]
                ]
                dst_mts.append(dst_ts_data)

            index_data.check_schema(output_schema)

        return {"output": dst_evset}


implementation_lib.register_operator_implementation(
    Resample, ResampleNumpyImplementation
)
