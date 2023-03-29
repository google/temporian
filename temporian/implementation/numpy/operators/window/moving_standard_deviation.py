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
from temporian.core.operators.window.moving_standard_deviation import (
    MovingStandardDeviationOperator,
)
from temporian.implementation.numpy.operators.window.base import (
    BaseWindowNumpyImplementation,
)


class MovingStandardDeviationNumpyImplementation(BaseWindowNumpyImplementation):
    """Numpy implementation of the moving standard deviation operator."""

    def __init__(self, operator: MovingStandardDeviationOperator) -> None:
        super().__init__(operator)

    def _apply_operation(self, values: np.array) -> np.array:
        """
        Calculates the moving standard deviation of the values in each row of
        the input array.

        The input array should have a shape (n, m), where 'n' is the length of
        the feature and 'm' is the size of the window. Each row represents a
        window of data points, with 'nan' values used for padding when the
        window size is  smaller than the number of data points in the time
        series. The function  computes the moving standard deviation for each
        row (window) by ignoring the 'nan' values.



        Args:
            values: A 2D NumPy array with shape (n, m) where each row represents
                a  window of data points. 'n' is the length of the feature, and
                'm' is the size of the window. The array can contain 'nan'
                values as padding.

        Returns:
            np.array: A 1D NumPy array with shape (n,) containing the moving
                    standard deviation for each row (window) in the input array.

        """
        return np.nanstd(values, axis=1)


implementation_lib.register_operator_implementation(
    MovingStandardDeviationOperator, MovingStandardDeviationNumpyImplementation
)
