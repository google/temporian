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

from functools import partial
from typing import Dict, Optional, Type

import apache_beam as beam

from temporian.beam import implementation_lib
from temporian.beam.operators.window.base import (
    BaseWindowBeamImplementation,
    _run_with_sampling,
    _run_without_sampling,
)
from temporian.beam.operators.base import (
    beam_eventset_map,
    beam_eventset_map_with_sampling,
)
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
    BeamIndexKey,
    FeatureItemValue,
)
from temporian.core.operators.window.base import BaseWindowOperator
from temporian.core.operators.window.moving_count import (
    MovingCountOperator,
)
from temporian.implementation.numpy.operators.window.base import (
    BaseWindowNumpyImplementation,
)
from temporian.implementation.numpy.operators.window.moving_count import (
    MovingCountNumpyImplementation,
)


class MovingCountBeamImplementation(BaseWindowBeamImplementation):
    def _implementation(self) -> Type[BaseWindowNumpyImplementation]:
        return MovingCountNumpyImplementation

    def call(
        self, input: BeamEventSet, sampling: Optional[BeamEventSet] = None
    ) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, BaseWindowOperator)

        numpy_implementation = self._implementation()(self.operator)

        if self.operator.has_sampling:
            assert sampling is not None
            # Run on first feature only since only timestamps are used
            input = (input[0],)

            output = beam_eventset_map_with_sampling(
                input,
                sampling,
                name=f"{self.operator}",
                fn=partial(_run_with_sampling, numpy_implementation),
            )

        else:
            # Run on first feature only since only timestamps are used
            input = (input[0],)

            output = beam_eventset_map(
                input,
                name=f"{self.operator}",
                fn=partial(_run_without_sampling, numpy_implementation),
            )

        return {"output": output}


implementation_lib.register_operator_implementation(
    MovingCountOperator, MovingCountBeamImplementation
)
