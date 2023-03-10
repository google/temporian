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

from temporian.implementation.pandas.operators import assign
from temporian.implementation.pandas.operators import select
from temporian.implementation.pandas.operators import sum
from temporian.implementation.pandas.operators.window import (
    simple_moving_average,
)

OPERATOR_IMPLEMENTATIONS = {
    "ASSIGN": assign.PandasAssignOperator,
    "SIMPLE_MOVING_AVERAGE": simple_moving_average.PandasSimpleMovingAverageOperator,
    "SELECT": select.PandasSelectOperator,
    "SUM": sum.PandasSumOperator,
}
