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
from typing import Union

import numpy as np

from temporian.implementation.numpy.operators.scalar.base import (
    BaseScalarNumpyImplementation,
)
from temporian.core.operators.scalar import (
    EqualScalarOperator,
    NotEqualScalarOperator,
    GreaterEqualScalarOperator,
    LessEqualScalarOperator,
    GreaterScalarOperator,
    LessScalarOperator,
)
from temporian.implementation.numpy import implementation_lib


class EqualScalarNumpyImplementation(BaseScalarNumpyImplementation):
    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        # Returns False if both NaNs
        return np.equal(feature, value)


class NotEqualScalarNumpyImplementation(BaseScalarNumpyImplementation):
    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        return np.not_equal(feature, value)


class GreaterEqualScalarNumpyImplementation(BaseScalarNumpyImplementation):
    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        return np.greater_equal(feature, value)


class LessEqualScalarNumpyImplementation(BaseScalarNumpyImplementation):
    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        return np.less_equal(feature, value)


class GreaterScalarNumpyImplementation(BaseScalarNumpyImplementation):
    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        return np.greater(feature, value)


class LessScalarNumpyImplementation(BaseScalarNumpyImplementation):
    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        return np.less(feature, value)


implementation_lib.register_operator_implementation(
    EqualScalarOperator, EqualScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    NotEqualScalarOperator, NotEqualScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    GreaterEqualScalarOperator, GreaterEqualScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    LessEqualScalarOperator, LessEqualScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    GreaterScalarOperator, GreaterScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    LessScalarOperator, LessScalarNumpyImplementation
)
