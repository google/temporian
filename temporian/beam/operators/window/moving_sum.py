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

from typing import Dict, Optional

import apache_beam as beam

from temporian.core.operators.window.moving_sum import (
    MovingSumOperator as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.implementation.numpy.operators.window.moving_sum import (
    MovingSumNumpyImplementation as CurrentOperatorImplementation,
)
from temporian.beam.operators.base import (
    BeamOperatorImplementation,
    beam_eventset_map,
    beam_eventset_map_with_sampling,
)
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
    BeamIndexKey,
    FeatureItemValue,
)
from temporian.implementation.numpy.operators.base import OperatorImplementation


class MovingSumBeamImplementation(BeamOperatorImplementation):
    def call(
        self, input: BeamEventSet, sampling: Optional[BeamEventSet] = None
    ) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        numpy_implementation = CurrentOperatorImplementation(self.operator)

        if self.operator.has_sampling:

            def _run_with_sampling(
                index: BeamIndexKey,
                feature: Optional[FeatureItemValue],
                sampling: FeatureItemValue,
                feature_idx: int,
            ) -> FeatureItem:
                sampling_timestamps, _ = sampling
                output_values = (
                    numpy_implementation.apply_feature_wise_with_sampling(
                        src_timestamps=(
                            feature[0] if feature is not None else None
                        ),
                        src_feature=feature[1] if feature is not None else None,
                        sampling_timestamps=sampling_timestamps,
                        feature_idx=feature_idx,
                    )
                )
                return index, (sampling_timestamps, output_values)

            assert sampling is not None
            output = beam_eventset_map_with_sampling(
                input,
                sampling,
                name=f"{self.operator}",
                fn=_run_with_sampling,
            )

        else:

            def _run_without_sampling(
                item: FeatureItem,
                feature_idx: int,
            ) -> FeatureItem:
                indexes, (timestamps, input_values) = item
                output_values = numpy_implementation.apply_feature_wise(
                    src_timestamps=timestamps,
                    src_feature=input_values,
                    feature_idx=feature_idx,
                )
                return indexes, (timestamps, output_values)

            output = beam_eventset_map(
                input, name=f"{self.operator}", fn=_run_without_sampling
            )

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, MovingSumBeamImplementation
)
