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


from temporian.core.operators.select import (
    SelectOperator as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.beam.operators.base import BeamOperatorImplementation
from temporian.beam.typing import BeamEventSet


class SelectBeamImplementation(BeamOperatorImplementation):
    def call(self, input: BeamEventSet) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        # Index of the features to keep in "input".
        src_feature_names = self.operator.inputs["input"].schema.feature_names()
        feature_idxs = set(
            [
                src_feature_names.index(feature_name)
                for feature_name in self.operator.feature_names
            ]
        )

        output = tuple([input[feature_idx] for feature_idx in feature_idxs])
        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, SelectBeamImplementation
)
