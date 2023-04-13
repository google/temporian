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

from typing import Any, Union
import numpy as np

from temporian.implementation.numpy import implementation_lib
from temporian.core.operators.boolean.equal_feature import EqualFeatureOperator
from temporian.core.operators.boolean.equal_scalar import EqualScalarOperator

from temporian.implementation.numpy.operators.boolean.base import (
    BaseBooleanNumpyImplementation,
)


class EqualNumpyImplementation(BaseBooleanNumpyImplementation):
    """Numpy implementation of the equal operator."""

    def __init__(
        self, operator: Union[EqualFeatureOperator, EqualScalarOperator]
    ) -> None:
        assert isinstance(operator, (EqualFeatureOperator, EqualScalarOperator))
        super().__init__(operator)

    def operation(
        self, feature_data: np.ndarray, value: Union[np.ndarray, Any]
    ) -> np.ndarray:
        return np.equal(feature_data, value)


implementation_lib.register_operator_implementation(
    EqualFeatureOperator, EqualNumpyImplementation
)

implementation_lib.register_operator_implementation(
    EqualScalarOperator, EqualNumpyImplementation
)
