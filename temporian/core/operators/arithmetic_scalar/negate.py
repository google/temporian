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

"""Negate operator class and public API function definition."""

from typing import List

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class NegateOperator(BaseArithmeticScalarOperator):
    def __init__(
        self,
        input: Node,
        value: int = -1,
        is_value_first: bool = False,
    ):
        super().__init__(input, value, is_value_first)

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "NEGATE"

    # this will be ignored because we will override output feature name
    @property
    def prefix(self) -> str:
        return ""

    # overriding feature name to be the same as the input feature
    def output_feature_name(self, feature_name: str) -> str:
        return feature_name

    # overriding checking for feature dtype to be the same as value dtype
    @property
    def ignore_value_dtype_checking(self) -> bool:
        return True

    @property
    def supported_value_dtypes(self) -> List[DType]:
        return [
            DType.INT32,
            DType.INT64,
        ]


operator_lib.register_operator(NegateOperator)


def negate(
    input: Node,
) -> Node:
    """Negates a node.

    Multiplies each item in each feature in `input` by -1.

    Args:
        input: Node to negate.

    Returns:
        Negated node.
    """
    return NegateOperator(
        input=input,
        value=-1,
    ).outputs["output"]
