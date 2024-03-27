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

import numpy as np
from typing import Dict, Optional, Any
import dataclasses

from temporian.core.operators.resample import (
    Resample as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.beam.operators.base import (
    BeamOperatorImplementation,
    beam_eventset_map_with_sampling,
)
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
    BeamIndexKey,
    FeatureItemValue,
    FeatureItemValue,
)
from temporian.implementation.numpy_cc.operators import operators_cc
from temporian.implementation.numpy.data.dtype_normalization import (
    tp_dtype_to_np_dtype,
)


@dataclasses.dataclass
class OutputSpec:
    missing_value: Any
    numpy_dtype: Any


class ResampleBeamImplementation(BeamOperatorImplementation):
    def call(
        self, input: BeamEventSet, sampling: BeamEventSet
    ) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        output_schema = self.operator.outputs["output"].schema
        output_missing_and_np_dtypes = [
            OutputSpec(
                missing_value=f.dtype.missing_value(),
                numpy_dtype=tp_dtype_to_np_dtype(f.dtype),
            )
            for f in output_schema.features
        ]

        def fun(
            index: BeamIndexKey,
            feature: Optional[FeatureItemValue],
            sampling: FeatureItemValue,
            feature_idx: int,
        ) -> FeatureItem:
            sampling_timestamps, _ = sampling

            output_spec = output_missing_and_np_dtypes[feature_idx]

            if feature is None:
                # No matching events to sample from
                sampled_features = np.full(
                    shape=len(sampling_timestamps),
                    fill_value=output_spec.missing_value,
                    dtype=output_spec.numpy_dtype,
                )
                return index, (sampling_timestamps, sampled_features)
            else:
                feature_timestamps, feature_values = feature
                assert feature_values is not None
                (
                    sampling_idxs,
                    first_valid_idx,
                ) = operators_cc.build_sampling_idxs(
                    feature_timestamps, sampling_timestamps
                )
                dst_ts_data = np.full(
                    shape=len(sampling_timestamps),
                    fill_value=output_spec.missing_value,
                    dtype=feature_values.dtype,
                )
                dst_ts_data[first_valid_idx:] = feature_values[
                    sampling_idxs[first_valid_idx:]
                ]
                return index, (sampling_timestamps, dst_ts_data)

        output = beam_eventset_map_with_sampling(
            input,
            sampling,
            name=f"{self.operator}",
            fn=fun,
        )

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, ResampleBeamImplementation
)
