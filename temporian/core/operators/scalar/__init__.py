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

"""Scalar operators."""

# pylint: disable=unused-import

from temporian.core.operators.scalar.arithmetic_scalar import (
    add_scalar,
    subtract_scalar,
    multiply_scalar,
    divide_scalar,
    floordiv_scalar,
    modulo_scalar,
    power_scalar,
    AddScalarOperator,
    SubtractScalarOperator,
    MultiplyScalarOperator,
    DivideScalarOperator,
    FloorDivScalarOperator,
    ModuloScalarOperator,
    PowerScalarOperator,
)

from temporian.core.operators.scalar.relational_scalar import (
    equal_scalar,
    not_equal_scalar,
    greater_equal_scalar,
    greater_scalar,
    less_equal_scalar,
    less_scalar,
    EqualScalarOperator,
    NotEqualScalarOperator,
    GreaterEqualScalarOperator,
    GreaterScalarOperator,
    LessEqualScalarOperator,
    LessScalarOperator,
)
