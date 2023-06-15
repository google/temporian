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

from temporian.core.operators.binary import (
    AddOperator,
    SubtractOperator,
    MultiplyOperator,
    DivideOperator,
    FloorDivOperator,
    ModuloOperator,
    PowerOperator,
)
from temporian.core.data.dtypes.dtype import DType
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.binary.base import (
    BaseBinaryNumpyImplementation,
)


class AddNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of add operator."""

    def __init__(self, operator: AddOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, AddOperator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return evset_1_feature + evset_2_feature


class SubtractNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the subtract operator."""

    def __init__(self, operator: SubtractOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, SubtractOperator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return evset_1_feature - evset_2_feature


class MultiplyNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the multiply operator."""

    def __init__(self, operator: MultiplyOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, MultiplyOperator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return evset_1_feature * evset_2_feature


class DivideNumpyImplementation(BaseBinaryNumpyImplementation):
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


class FloorDivNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the floordiv operator."""

    def __init__(self, operator: FloorDivOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, FloorDivOperator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return evset_1_feature // evset_2_feature


class ModuloNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the modulo operator."""

    def __init__(self, operator: ModuloOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, ModuloOperator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return evset_1_feature % evset_2_feature


class PowerNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the power operator."""

    def __init__(self, operator: PowerOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, PowerOperator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return evset_1_feature**evset_2_feature


implementation_lib.register_operator_implementation(
    AddOperator, AddNumpyImplementation
)
implementation_lib.register_operator_implementation(
    SubtractOperator, SubtractNumpyImplementation
)
implementation_lib.register_operator_implementation(
    MultiplyOperator, MultiplyNumpyImplementation
)
implementation_lib.register_operator_implementation(
    DivideOperator, DivideNumpyImplementation
)
implementation_lib.register_operator_implementation(
    FloorDivOperator, FloorDivNumpyImplementation
)
implementation_lib.register_operator_implementation(
    ModuloOperator, ModuloNumpyImplementation
)
implementation_lib.register_operator_implementation(
    PowerOperator, PowerNumpyImplementation
)
