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
    LogicalAndOperator,
    LogicalOrOperator,
    LogicalXorOperator,
)
from temporian.implementation.numpy import implementation_lib


class LogicalAndNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the logical AND operator."""

    def __init__(self, operator: LogicalAndOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self,
        evset_1_feature: np.ndarray,
        evset_2_feature: np.ndarray,
        dtype: DType,
    ) -> np.ndarray:
        return np.logical_and(evset_1_feature.data, evset_2_feature.data)


class LogicalOrNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the logical OR operator."""

    def __init__(self, operator: LogicalOrOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self,
        evset_1_feature: np.ndarray,
        evset_2_feature: np.ndarray,
        dtype: DType,
    ) -> np.ndarray:
        return np.logical_or(evset_1_feature.data, evset_2_feature.data)


class LogicalXorNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the logical XOR operator."""

    def __init__(self, operator: LogicalXorOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self,
        evset_1_feature: np.ndarray,
        evset_2_feature: np.ndarray,
        dtype: DType,
    ) -> np.ndarray:
        return np.logical_xor(evset_1_feature.data, evset_2_feature.data)


implementation_lib.register_operator_implementation(
    LogicalAndOperator, LogicalAndNumpyImplementation
)
implementation_lib.register_operator_implementation(
    LogicalOrOperator, LogicalOrNumpyImplementation
)
implementation_lib.register_operator_implementation(
    LogicalXorOperator, LogicalXorNumpyImplementation
)
