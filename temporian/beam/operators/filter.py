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
from typing import Dict, Optional


from temporian.core.operators.filter import (
    FilterOperator as CurrentOperator,
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


class FilterBeamImplementation(BeamOperatorImplementation):
    def call(
        self, input: BeamEventSet, condition: BeamEventSet
    ) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        def fun(
            index: BeamIndexKey,
            feature: Optional[FeatureItemValue],
            condition: FeatureItemValue,
            feature_idx: int,
        ) -> FeatureItem:
            condition_timestamps, mask = condition
            assert feature is not None
            feature_timestamps, feature_values = feature
            if feature_values is None:
                return index, (condition_timestamps[mask], None)
            else:
                assert np.array_equal(feature_timestamps, condition_timestamps)
                return index, (condition_timestamps[mask], feature_values[mask])

        output = beam_eventset_map_with_sampling(
            input,
            condition,
            name=f"{self.operator}",
            fn=fun,
        )

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, FilterBeamImplementation
)
