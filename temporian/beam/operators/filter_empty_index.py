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

import apache_beam as beam

from temporian.core.operators.filter_empty_index import (
    FilterEmptyIndex as CurrentOperator,
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


class FilterEmptyIndexBeamImplementation(BeamOperatorImplementation):
    def call(self, input: BeamEventSet) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        def fun(item: FeatureItem) -> bool:
            _, (timestamps, _) = item
            return len(timestamps) > 0

        def apply(idx, item):
            return (
                item
                | f"Map on feature #{idx} {self.operator}" >> beam.Filter(fun)
            )

        output = tuple([apply(idx, item) for idx, item in enumerate(input)])

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, FilterEmptyIndexBeamImplementation
)
