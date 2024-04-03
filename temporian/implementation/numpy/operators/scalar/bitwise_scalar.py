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
from temporian.implementation.numpy.operators.scalar.base import (
    BaseScalarNumpyImplementation,
)
from temporian.core.operators.scalar import (
    BitwiseAndScalarOperator,
    BitwiseOrScalarOperator,
    BitwiseXorScalarOperator,
)
from temporian.implementation.numpy import implementation_lib


class BitwiseAndScalarNumpyImplementation(BaseScalarNumpyImplementation):
    """Numpy implementation of the bitwise and scalar operator."""

    def _do_operation(
        self,
        feature: np.ndarray,
        value: int,
        dtype: DType,
    ) -> np.ndarray:
        return np.bitwise_and(feature.data, value)


class BitwiseOrScalarNumpyImplementation(BaseScalarNumpyImplementation):
    """Numpy implementation of the bitwise or scalar operator."""

    def _do_operation(
        self,
        feature: np.ndarray,
        value: int,
        dtype: DType,
    ) -> np.ndarray:
        return np.bitwise_or(feature.data, value)


class BitwiseXorScalarNumpyImplementation(BaseScalarNumpyImplementation):
    """Numpy implementation of the bitwise xor scalar operator."""

    def _do_operation(
        self,
        feature: np.ndarray,
        value: int,
        dtype: DType,
    ) -> np.ndarray:
        return np.bitwise_xor(feature.data, value)


implementation_lib.register_operator_implementation(
    BitwiseAndScalarOperator, BitwiseAndScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    BitwiseOrScalarOperator, BitwiseOrScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    BitwiseXorScalarOperator, BitwiseXorScalarNumpyImplementation
)
