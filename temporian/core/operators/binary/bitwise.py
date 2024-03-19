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

"""Binary logic operators classes and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import EventSetNode
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.binary.base import BaseBinaryOperator
from temporian.core.typing import EventSetOrNode


class BaseBitwiseOperator(BaseBinaryOperator):
    OP_NAME = ""

    def __init__(self, input_1: EventSetNode, input_2: EventSetNode):
        super().__init__(input_1, input_2)

        # Check that all features are Integer
        # Note: Assuming that input_1 and input_2 features have the same dtype
        for feature in input_1.schema.features:
            if feature.dtype != DType.INT32 and feature.dtype != DType.INT64:
                raise ValueError(
                    "Bitwise operators only support INT32 or INT64 types, but feature"
                    f" {feature.name} has dtype {feature.dtype}"
                )

    def output_feature_dtype(
        self, feature_1: FeatureSchema, feature_2: FeatureSchema
    ) -> DType:
        if feature_1.dtype == DType.INT64 or feature_2.dtype == DType.INT64:
            return DType.INT64
        else:
            return DType.INT32

    @classmethod
    def operator_def_key(cls) -> str:
        return cls.OP_NAME.upper()

    @property
    def prefix(self) -> str:
        return self.OP_NAME.lower()


class BitwiseAndOperator(BaseBitwiseOperator):
    OP_NAME = "bitwise_and"


class BitwiseOrOperator(BaseBitwiseOperator):
    OP_NAME = "bitwise_or"


class BitwiseXorOperator(BaseBitwiseOperator):
    OP_NAME = "bitwise_xor"


@compile
def bitwise_and(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_1, EventSetNode)
    assert isinstance(input_2, EventSetNode)

    return BitwiseAndOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def bitwise_or(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_1, EventSetNode)
    assert isinstance(input_2, EventSetNode)

    return BitwiseOrOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def bitwise_xor(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input_1, EventSetNode)
    assert isinstance(input_2, EventSetNode)

    return BitwiseXorOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


operator_lib.register_operator(BitwiseAndOperator)
operator_lib.register_operator(BitwiseOrOperator)
operator_lib.register_operator(BitwiseXorOperator)
