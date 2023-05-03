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

from temporian.core.data.dtype import DType
from temporian.core.operators.arithmetic import DivideOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.arithmetic.base import (
    BaseArithmeticNumpyImplementation,
)


class DivideNumpyImplementation(BaseArithmeticNumpyImplementation):
    """Numpy implementation of the divide operator."""

    def __init__(self, operator: DivideOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, DivideOperator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        if evset_1_feature.dtype in [DType.INT32, DType.INT64]:
            raise ValueError(
                "Cannot use the divide operator on feature "
                f"{evset_1_feature} of type {evset_1_feature.dtype.type}. "
                "Cast to a floating point type or use "
                "floordiv operator (//) instead, on these integer types."
            )
        return evset_1_feature / evset_2_feature


implementation_lib.register_operator_implementation(
    DivideOperator, DivideNumpyImplementation
)
