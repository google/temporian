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

"""Event/scalar relational operators classes and public API definitions."""

from typing import Union, List

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.operators.scalar.base import (
    BaseScalarOperator,
)


class RelationalScalarOperator(BaseScalarOperator):
    def output_feature_dtype(self, feature: Feature) -> DType:
        # override parent method to always return BOOLEAN features
        return DType.BOOLEAN

    @property
    def supported_value_dtypes(self) -> List[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
            DType.BOOLEAN,
            DType.STRING,
        ]


class EqualScalarOperator(RelationalScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "EQUAL_SCALAR"


class NotEqualScalarOperator(RelationalScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "NOT_EQUAL_SCALAR"


class GreaterEqualScalarOperator(RelationalScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "GREATER_EQUAL_SCALAR"


class LessEqualScalarOperator(RelationalScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "LESS_EQUAL_SCALAR"


class GreaterScalarOperator(RelationalScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "GREATER_SCALAR"


class LessScalarOperator(RelationalScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "LESS_SCALAR"


def equal_scalar(
    input: Node,
    value: Union[float, int, str, bool],
) -> Node:
    """Checks for equality between a node and a scalar element-wise.

    Each item in each feature in `input` is compared to `value`.

    Args:
        input: Node to compare the value to.
        value: Scalar value to compare to the input.

    Returns:
        Node containing the result of the comparison.
    """
    return EqualScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


def not_equal_scalar(
    input: Node,
    value: Union[float, int, str, bool],
) -> Node:
    """Checks for differences between a node and a scalar element-wise.

    Each item in each feature in `input` is compared to `value`.

    Args:
        input: Node to compare the value to.
        value: Scalar value to compare to the input.

    Returns:
        Node containing the result of the comparison.
    """
    return NotEqualScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


def greater_equal_scalar(
    input: Node,
    value: Union[float, int, str, bool],
) -> Node:
    """Check if the input node is greater or equal than a scalar element-wise.

    Each item in each feature in `input` is compared to `value`.

    Args:
        input: Node to compare the value to.
        value: Scalar value to compare to the input.

    Returns:
        Node containing the result of the comparison.
    """
    return GreaterEqualScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


def less_equal_scalar(
    input: Node,
    value: Union[float, int, str, bool],
) -> Node:
    """Check if the input node is less or equal than a scalar element-wise.

    Each item in each feature in `input` is compared to `value`.

    Args:
        input: Node to compare the value to.
        value: Scalar value to compare to the input.

    Returns:
        Node containing the result of the comparison.
    """
    return LessEqualScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


def greater_scalar(
    input: Node,
    value: Union[float, int, str, bool],
) -> Node:
    """Check if the input node is greater than a scalar element-wise.

    Each item in each feature in `input` is compared to `value`.

    Args:
        input: Node to compare the value to.
        value: Scalar value to compare to the input.

    Returns:
        Node containing the result of the comparison.
    """
    return GreaterScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


def less_scalar(
    input: Node,
    value: Union[float, int, str, bool],
) -> Node:
    """Check if the input node is less than a scalar element-wise.

    Each item in each feature in `input` is compared to `value`.

    Args:
        input: Node to compare the value to.
        value: Scalar value to compare to the input.

    Returns:
        Node containing the result of the comparison.
    """
    return LessScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


operator_lib.register_operator(EqualScalarOperator)
operator_lib.register_operator(NotEqualScalarOperator)
operator_lib.register_operator(GreaterEqualScalarOperator)
operator_lib.register_operator(LessEqualScalarOperator)
operator_lib.register_operator(GreaterScalarOperator)
operator_lib.register_operator(LessScalarOperator)
