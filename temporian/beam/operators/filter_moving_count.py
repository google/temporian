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


from temporian.core.operators.filter_moving_count import (
    FilterMaxMovingCount as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.beam.operators.base import (
    BeamOperatorImplementation,
    beam_eventset_flatmap,
)
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
)
from temporian.implementation.numpy_cc.operators import operators_cc


class FilterMaxMovingCountBeamImplementation(BeamOperatorImplementation):
    def call(self, input: BeamEventSet) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        window_length = self.operator.window_length

        def fun(item: FeatureItem, feature_idx: int):
            indexes, (timestamps, feature) = item

            sampling_idx = operators_cc.filter_moving_count(
                timestamps,
                window_length=window_length,
            )
            new_timestamps = timestamps[sampling_idx]
            if feature is None:
                new_features = None
            else:
                new_features = feature[sampling_idx]
            yield indexes, (new_timestamps, new_features)

        output = beam_eventset_flatmap(
            input,
            name=f"{self.operator}",
            fn=fun,
        )

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, FilterMaxMovingCountBeamImplementation
)
