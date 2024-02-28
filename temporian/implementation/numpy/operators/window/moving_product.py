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


from temporian.core.operators.window.moving_product import (
    MovingProductOperator,
)
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.window.base import (
    BaseWindowNumpyImplementation,
)
from temporian.implementation.numpy_cc.operators import operators_cc


class MovingProductNumpyImplementation(BaseWindowNumpyImplementation):
    """Numpy implementation of the moving product operator."""

    def _implementation(self):
        return operators_cc.moving_product


implementation_lib.register_operator_implementation(
    MovingProductOperator, MovingProductNumpyImplementation
)
