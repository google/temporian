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
from temporian.implementation.numpy.operators.binary.base import (
    BaseBinaryNumpyImplementation,
)
from temporian.core.operators.binary import (
    BitwiseAndOperator,
    BitwiseOrOperator,
    BitwiseXorOperator,
)
from temporian.implementation.numpy import implementation_lib


class BitwiseAndNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the bitwise AND operator."""

    def __init__(self, operator: BitwiseAndOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self,
        evset_1_feature: np.ndarray,
        evset_2_feature: np.ndarray,
        dtype: DType,
    ) -> np.ndarray:
        return np.bitwise_and(evset_1_feature.data, evset_2_feature.data)


class BitwiseOrNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the bitwise OR operator."""

    def __init__(self, operator: BitwiseOrOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self,
        evset_1_feature: np.ndarray,
        evset_2_feature: np.ndarray,
        dtype: DType,
    ) -> np.ndarray:
        return np.bitwise_or(evset_1_feature.data, evset_2_feature.data)


class BitwiseXorNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the bitwise XOR operator."""

    def __init__(self, operator: BitwiseXorOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self,
        evset_1_feature: np.ndarray,
        evset_2_feature: np.ndarray,
        dtype: DType,
    ) -> np.ndarray:
        return np.bitwise_xor(evset_1_feature.data, evset_2_feature.data)


implementation_lib.register_operator_implementation(
    BitwiseAndOperator, BitwiseAndNumpyImplementation
)
implementation_lib.register_operator_implementation(
    BitwiseOrOperator, BitwiseOrNumpyImplementation
)
implementation_lib.register_operator_implementation(
    BitwiseXorOperator, BitwiseXorNumpyImplementation
)
