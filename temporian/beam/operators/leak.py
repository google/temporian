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


from temporian.core.operators.leak import (
    LeakOperator as CurrentOperator,
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


class LeakBeamImplementation(BeamOperatorImplementation):
    def call(self, input: BeamEventSet) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        duration = self.operator.duration

        def fun(item: FeatureItem, idx: int) -> FeatureItem:
            indexes, (timestamps, values) = item
            timestamps = timestamps - duration
            return indexes, (timestamps, values)

        output = beam_eventset_map(
            input,
            name=f"{self.operator}",
            fn=fun,
        )

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, LeakBeamImplementation
)
