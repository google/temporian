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


from temporian.core.operators.cast import (
    CastOperator as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.beam.operators.base import (
    BeamOperatorImplementation,
    beam_eventset_map,
)
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
)
from temporian.implementation.numpy.data.dtype_normalization import (
    tp_dtype_to_np_dtype,
)


class CastBeamImplementation(BeamOperatorImplementation):
    def call(self, input: BeamEventSet) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        if self.operator.is_noop:
            return {"output": input}

        # Numpy output dtype for each feature.
        np_dtypes = [
            tp_dtype_to_np_dtype(tp_dtype) for tp_dtype in self.operator.dtypes
        ]

        def fun(item: FeatureItem, feature_idx: int) -> FeatureItem:
            indexes, (timestamps, values) = item
            values = values.astype(np_dtypes[feature_idx])
            return indexes, (timestamps, values)

        output = beam_eventset_map(
            input,
            name=f"{self.operator}",
            fn=fun,
        )

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, CastBeamImplementation
)
