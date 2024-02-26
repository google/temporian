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

"""Unary operators (e.g: ~, isnan, abs) and public API definitions."""

from abc import abstractmethod
from typing import List

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_existing_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb


class BaseUnaryOperator(Operator):
    def __init__(
        self,
        input: EventSetNode,
    ):
        super().__init__()

        # Check input
        if not isinstance(input, EventSetNode):
            raise TypeError(
                f"Input must be of type EventSetNode but got {type(input)}"
            )

        for feature in input.schema.features:
            if feature.dtype not in self.allowed_dtypes():
                raise ValueError(
                    "DTypes supported by the operator:"
                    f" {self.allowed_dtypes()}. Got feature {feature.name} with"
                    f" dtype {feature.dtype}."
                )

        self.add_input("input", input)

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=[
                    (feature.name, self.get_output_dtype(feature.dtype))
                    for feature in input.schema.features
                ],
                sampling_node=input,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key=cls.op_key_definition(),
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="input"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @classmethod
    @abstractmethod
    def op_key_definition(cls) -> str:
        """Gets the op. key used for serialization in build_op_definition."""

    @classmethod
    @abstractmethod
    def allowed_dtypes(cls) -> List[DType]:
        """Gets the dtypes that should work with this operator."""

    @classmethod
    @abstractmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        """Gets the output DType from the input feature DType."""


class InvertOperator(BaseUnaryOperator):
    @classmethod
    def op_key_definition(cls) -> str:
        return "INVERT"

    @classmethod
    def allowed_dtypes(cls) -> List[DType]:
        return [DType.BOOLEAN]

    @classmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        return DType.BOOLEAN


class IsNanOperator(BaseUnaryOperator):
    @classmethod
    def op_key_definition(cls) -> str:
        return "IS_NAN"

    @classmethod
    def allowed_dtypes(cls) -> List[DType]:
        return [
            DType.BOOLEAN,
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]

    @classmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        return DType.BOOLEAN


class NotNanOperator(BaseUnaryOperator):
    @classmethod
    def op_key_definition(cls) -> str:
        return "NOT_NAN"

    @classmethod
    def allowed_dtypes(cls) -> List[DType]:
        return [
            DType.BOOLEAN,
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]

    @classmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        return DType.BOOLEAN


class AbsOperator(BaseUnaryOperator):
    @classmethod
    def op_key_definition(cls) -> str:
        return "ABS"

    @classmethod
    def allowed_dtypes(cls) -> List[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]

    @classmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        return feature_dtype


class LogOperator(BaseUnaryOperator):
    @classmethod
    def op_key_definition(cls) -> str:
        return "LOG"

    @classmethod
    def allowed_dtypes(cls) -> List[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
        ]

    @classmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        return feature_dtype


class RoundOperator(BaseUnaryOperator):
    @classmethod
    def op_key_definition(cls) -> str:
        return "ROUND"

    @classmethod
    def allowed_dtypes(cls) -> List[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
        ]

    @classmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        return feature_dtype


operator_lib.register_operator(InvertOperator)
operator_lib.register_operator(IsNanOperator)
operator_lib.register_operator(NotNanOperator)
operator_lib.register_operator(AbsOperator)
operator_lib.register_operator(LogOperator)
operator_lib.register_operator(RoundOperator)


@compile
def invert(
    input: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return InvertOperator(
        input=input,
    ).outputs["output"]


@compile
def isnan(
    input: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return IsNanOperator(
        input=input,
    ).outputs["output"]


@compile
def notnan(
    input: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return NotNanOperator(
        input=input,
    ).outputs["output"]


@compile
def abs(
    input: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return AbsOperator(
        input=input,
    ).outputs["output"]


@compile
def log(
    input: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return LogOperator(
        input=input,
    ).outputs["output"]


@compile
def round(
    input: EventSetOrNode,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return RoundOperator(
        input=input,
    ).outputs["output"]
