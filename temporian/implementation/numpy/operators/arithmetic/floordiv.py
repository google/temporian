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

from temporian.core.operators.arithmetic import FloorDivOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.arithmetic.base import (
    BaseArithmeticNumpyImplementation,
)


class FloorDivideNumpyImplementation(BaseArithmeticNumpyImplementation):
    """Actual numpy implementation to divide of first event by the second one
    and take the result floor"""

    def __init__(self, operator: FloorDivOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, FloorDivOperator)

    def _do_operation(
        self, event_1_feature: np.ndarray, event_2_feature: np.ndarray
    ) -> np.ndarray:
        return event_1_feature // event_2_feature


implementation_lib.register_operator_implementation(
    FloorDivOperator, FloorDivideNumpyImplementation
)
