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

from temporian.core.operators.window.moving_sum import (
    MovingSumOperator as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.implementation.numpy.operators.window.moving_sum import (
    MovingSumNumpyImplementation as CurrentOperatorImplementation,
)
from temporian.beam.operators.base import BeamOperatorImplementation
from temporian.beam.io import BeamEventSet, PColBeamEventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


def _run_item(pipe: BeamEventSet, imp: OperatorImplementation):
    indexes, (timestamps, input_values) = pipe
    output_values = imp.apply_feature_wise(
        src_timestamps=timestamps,
        src_feature=input_values,
    )
    return indexes, (timestamps, output_values)


class MovingSumBeamImplementation(BeamOperatorImplementation):
    def call(self, input: PColBeamEventSet) -> Dict[str, PColBeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        assert not self.operator.has_sampling

        numpy_implementation = CurrentOperatorImplementation(self.operator)

        output = input | f"Apply operator {self.operator}" >> beam.Map(
            _run_item, numpy_implementation
        )

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, MovingSumBeamImplementation
)
