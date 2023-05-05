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
    AddScalarOperator,
    SubtractScalarOperator,
    MultiplyScalarOperator,
    DivideScalarOperator,
    FloorDivScalarOperator,
)
from temporian.implementation.numpy import implementation_lib


class AddScalarNumpyImplementation(BaseScalarNumpyImplementation):
    """Numpy implementation of the add scalar operator."""

    def __init__(self, operator: AddScalarOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        return feature + value


class SubtractScalarNumpyImplementation(BaseScalarNumpyImplementation):
    """Numpy implementation of the subtract scalar operator."""

    def __init__(self, operator: SubtractScalarOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        if self._operator.is_value_first:
            return value - feature

        return feature - value


class MultiplyScalarNumpyImplementation(BaseScalarNumpyImplementation):
    """Numpy implementation of the multiply scalar operator."""

    def __init__(self, operator: MultiplyScalarOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        return feature * value


class DivideScalarNumpyImplementation(BaseScalarNumpyImplementation):
    """Numpy implementation of the divide scalar operator."""

    def __init__(self, operator: DivideScalarOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        if self._operator.is_value_first:
            return value / feature

        return feature / value


class FloorDivideScalarNumpyImplementation(BaseScalarNumpyImplementation):
    """Numpy implementation of the floordiv scalar operator."""

    def __init__(self, operator: FloorDivScalarOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        if self._operator.is_value_first:
            return value // feature

        return feature // value


implementation_lib.register_operator_implementation(
    AddScalarOperator, AddScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    SubtractScalarOperator, SubtractScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    MultiplyScalarOperator, MultiplyScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    DivideScalarOperator, DivideScalarNumpyImplementation
)
implementation_lib.register_operator_implementation(
    FloorDivScalarOperator, FloorDivideScalarNumpyImplementation
)
