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
from temporian.implementation.numpy.operators.binary.base import (
    BaseBinaryNumpyImplementation,
)
from temporian.core.operators.binary import (
    EqualOperator,
    NotEqualOperator,
    GreaterOperator,
    GreaterEqualOperator,
    LessOperator,
    LessEqualOperator,
)
from temporian.implementation.numpy import implementation_lib


class EqualNumpyImplementation(BaseBinaryNumpyImplementation):
    """Numpy implementation of the equal operator."""

    def __init__(self, operator: EqualOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        # returns False on both NaNs
        return np.equal(evset_1_feature.data, evset_2_feature.data)


class NotEqualNumpyImplementation(BaseBinaryNumpyImplementation):
    def __init__(self, operator: EqualOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return np.not_equal(evset_1_feature.data, evset_2_feature.data)


class GreaterNumpyImplementation(BaseBinaryNumpyImplementation):
    def __init__(self, operator: EqualOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return np.greater(evset_1_feature.data, evset_2_feature.data)


class GreaterEqualNumpyImplementation(BaseBinaryNumpyImplementation):
    def __init__(self, operator: EqualOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return np.greater_equal(evset_1_feature.data, evset_2_feature.data)


class LessNumpyImplementation(BaseBinaryNumpyImplementation):
    def __init__(self, operator: EqualOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return np.less(evset_1_feature.data, evset_2_feature.data)


class LessEqualNumpyImplementation(BaseBinaryNumpyImplementation):
    def __init__(self, operator: EqualOperator) -> None:
        super().__init__(operator)

    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        return np.less_equal(evset_1_feature.data, evset_2_feature.data)


implementation_lib.register_operator_implementation(
    EqualOperator, EqualNumpyImplementation
)
implementation_lib.register_operator_implementation(
    NotEqualOperator, NotEqualNumpyImplementation
)
implementation_lib.register_operator_implementation(
    GreaterEqualOperator, GreaterEqualNumpyImplementation
)
implementation_lib.register_operator_implementation(
    LessEqualOperator, LessEqualNumpyImplementation
)
implementation_lib.register_operator_implementation(
    LessOperator, LessNumpyImplementation
)
implementation_lib.register_operator_implementation(
    GreaterOperator, GreaterNumpyImplementation
)
