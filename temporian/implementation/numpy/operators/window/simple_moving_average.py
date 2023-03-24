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

from temporian.implementation.numpy import implementation_lib
from temporian.core.operators.window.simple_moving_average import (
    SimpleMovingAverageOperator,
)
from temporian.implementation.numpy.operators.window.base import (
    BaseWindowNumpyImplementation,
)


class SimpleMovingAverageNumpyImplementation(BaseWindowNumpyImplementation):
    """Numpy implementation of the simple moving average operator."""

    def __init__(self, operator: SimpleMovingAverageOperator) -> None:
        super().__init__(operator)

    def _calculate_window_operation(self, values: np.array) -> np.array:
        """Calculates the simple moving average of the values in the window.

        Args:
            values: The values in the window.

        Returns:
            The simple moving average of the values in the window.
        """
        return np.nanmean(values, axis=1)


implementation_lib.register_operator_implementation(
    SimpleMovingAverageOperator, SimpleMovingAverageNumpyImplementation
)
