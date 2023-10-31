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
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import EventSetNode
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.scalar.base import (
    BaseScalarOperator,
)
from temporian.core.typing import EventSetOrNode


class RelationalScalarOperator(BaseScalarOperator):
    DEF_KEY = ""

    def output_feature_dtype(self, feature: FeatureSchema) -> DType:
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

    @classmethod
    def operator_def_key(cls) -> str:
        return cls.DEF_KEY


class EqualScalarOperator(RelationalScalarOperator):
    DEF_KEY = "EQUAL_SCALAR"


class NotEqualScalarOperator(RelationalScalarOperator):
    DEF_KEY = "NOT_EQUAL_SCALAR"


class GreaterEqualScalarOperator(RelationalScalarOperator):
    DEF_KEY = "GREATER_EQUAL_SCALAR"


class LessEqualScalarOperator(RelationalScalarOperator):
    DEF_KEY = "LESS_EQUAL_SCALAR"


class GreaterScalarOperator(RelationalScalarOperator):
    DEF_KEY = "GREATER_SCALAR"


class LessScalarOperator(RelationalScalarOperator):
    DEF_KEY = "LESS_SCALAR"


@compile
def equal_scalar(
    input: EventSetOrNode,
    value: Union[float, int, str, bool, bytes],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return EqualScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


@compile
def not_equal_scalar(
    input: EventSetOrNode,
    value: Union[float, int, str, bytes, bool],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return NotEqualScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


@compile
def greater_equal_scalar(
    input: EventSetOrNode,
    value: Union[float, int, str, bytes, bool],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return GreaterEqualScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


@compile
def less_equal_scalar(
    input: EventSetOrNode,
    value: Union[float, int, str, bytes, bool],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return LessEqualScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


@compile
def greater_scalar(
    input: EventSetOrNode,
    value: Union[float, int, str, bytes, bool],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return GreaterScalarOperator(
        input=input,
        value=value,
    ).outputs["output"]


@compile
def less_scalar(
    input: EventSetOrNode,
    value: Union[float, int, str, bytes, bool],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

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
