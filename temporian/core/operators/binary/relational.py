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

"""Binary relational operators classes and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import EventSetNode
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.binary.base import BaseBinaryOperator
from temporian.core.typing import EventSetOrNode


class BaseRelationalOperator(BaseBinaryOperator):
    DEF_KEY = ""
    PREFIX = ""

    @classmethod
    def operator_def_key(cls) -> str:
        return cls.DEF_KEY

    @property
    def prefix(self) -> str:
        return self.PREFIX

    # override parent dtype method
    def output_feature_dtype(
        self, feature_1: FeatureSchema, feature_2: FeatureSchema
    ) -> DType:
        return DType.BOOLEAN


class EqualOperator(BaseRelationalOperator):
    DEF_KEY = "EQUAL"
    PREFIX = "eq"


class NotEqualOperator(BaseRelationalOperator):
    DEF_KEY = "NOT_EQUAL"
    PREFIX = "ne"


class GreaterOperator(BaseRelationalOperator):
    DEF_KEY = "GREATER"
    PREFIX = "gt"


class GreaterEqualOperator(BaseRelationalOperator):
    DEF_KEY = "GREATER_EQUAL"
    PREFIX = "ge"


class LessOperator(BaseRelationalOperator):
    DEF_KEY = "LESS"
    PREFIX = "lt"


class LessEqualOperator(BaseRelationalOperator):
    DEF_KEY = "LESS_EQUAL"
    PREFIX = "le"


@compile
def equal(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_1, EventSetNode)
    assert isinstance(input_2, EventSetNode)

    return EqualOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def not_equal(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_1, EventSetNode)
    assert isinstance(input_2, EventSetNode)

    return NotEqualOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def greater(
    input_left: EventSetOrNode,
    input_right: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_left, EventSetNode)
    assert isinstance(input_right, EventSetNode)

    return GreaterOperator(
        input_1=input_left,
        input_2=input_right,
    ).outputs["output"]


@compile
def greater_equal(
    input_left: EventSetOrNode,
    input_right: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_left, EventSetNode)
    assert isinstance(input_right, EventSetNode)

    return GreaterEqualOperator(
        input_1=input_left,
        input_2=input_right,
    ).outputs["output"]


@compile
def less(
    input_left: EventSetOrNode,
    input_right: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_left, EventSetNode)
    assert isinstance(input_right, EventSetNode)

    return LessOperator(
        input_1=input_left,
        input_2=input_right,
    ).outputs["output"]


@compile
def less_equal(
    input_left: EventSetOrNode,
    input_right: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_left, EventSetNode)
    assert isinstance(input_right, EventSetNode)

    return LessEqualOperator(
        input_1=input_left,
        input_2=input_right,
    ).outputs["output"]


operator_lib.register_operator(EqualOperator)
operator_lib.register_operator(NotEqualOperator)
operator_lib.register_operator(GreaterOperator)
operator_lib.register_operator(GreaterEqualOperator)
operator_lib.register_operator(LessOperator)
operator_lib.register_operator(LessEqualOperator)
