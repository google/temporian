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

"""Unary operators (~, isnan, abs) and public API definitions"""

from abc import abstractmethod

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseUnaryOperator(Operator):
    def __init__(
        self,
        input: Node,
    ):
        super().__init__()

        # Check input
        if not isinstance(input, Node):
            raise TypeError(f"Input must be of type Node but got {type(input)}")

        for feature in input.features:
            if feature.dtype not in self.allowed_dtypes:
                raise ValueError(
                    f"DTypes supported by the operator: {self.allowed_dtypes}."
                    f" Got feature {feature.name} with dtype {feature.dtype}."
                )

        # inputs
        self.add_input("input", input)

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=feature.name,
                dtype=self.get_output_dtype(feature.dtype),
                sampling=input.sampling,
                creator=self,
            )
            for feature in input.features
        ]

        self.add_output(
            "output",
            Node(
                features=output_features,
                sampling=input.sampling,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key=cls.op_key_definition,
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="input"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @classmethod
    @property
    @abstractmethod
    def op_key_definition(cls) -> str:
        """
        Get the operator key used for serialization (build_op_definition)
        """

    @classmethod
    @property
    @abstractmethod
    def allowed_dtypes(cls) -> list[DType]:
        """
        Get the dtypes that should work with this operator
        """

    @classmethod
    @abstractmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        """
        Get the output DType from the corresponding feature DType
        """


class InvertOperator(BaseUnaryOperator):
    @classmethod
    @property
    def op_key_definition(cls) -> str:
        return "INVERT"

    @classmethod
    @property
    def allowed_dtypes(cls) -> list[DType]:
        return [DType.BOOLEAN]

    @classmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        return DType.BOOLEAN


class IsNanOperator(BaseUnaryOperator):
    @classmethod
    @property
    def op_key_definition(cls) -> str:
        return "IS_NAN"

    @classmethod
    @property
    def allowed_dtypes(cls) -> list[DType]:
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
    @property
    def op_key_definition(cls) -> str:
        return "NOT_NAN"

    @classmethod
    @property
    def allowed_dtypes(cls) -> list[DType]:
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
    @property
    def op_key_definition(cls) -> str:
        return "ABS"

    @classmethod
    @property
    def allowed_dtypes(cls) -> list[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]

    @classmethod
    def get_output_dtype(cls, feature_dtype: DType) -> DType:
        return feature_dtype


operator_lib.register_operator(InvertOperator)
operator_lib.register_operator(IsNanOperator)
operator_lib.register_operator(NotNanOperator)
operator_lib.register_operator(AbsOperator)


def invert(
    input: Node,
) -> Node:
    """Invert a boolean node (~node).
    Does not work on integers, they should be cast to BOOLEAN beforehand.
    Swaps False<->True

    Args:
        input: Node to invert.

    Returns:
        Negated node.
    """
    return InvertOperator(
        input=input,
    ).outputs["output"]


def isnan(
    input: Node,
) -> Node:
    """Get boolean features, True in the positions where
    there's a NaN.
    Note that for INT and BOOLEAN this will
    always be False since those types don't support NaNs.
    It only makes actual sense to use on FLOAT features.

    Args:
        input: Node to check for NaNs.

    Returns:
        Node with BOOLEAN features.
    """
    return IsNanOperator(
        input=input,
    ).outputs["output"]


def notnan(
    input: Node,
) -> Node:
    """Get boolean features, True in the positions that are not NaN.
    Equivalent to invert(isnan()). Note that for INT and BOOLEAN this will
    always be True since those types don't support NaNs.
    It only makes actual sense to use on FLOAT features.

    Args:
        input: Node to check for NaNs.

    Returns:
        Node with BOOLEAN features.
    """
    return NotNanOperator(
        input=input,
    ).outputs["output"]


def abs(
    input: Node,
) -> Node:
    """Get the absolute value of the features.

    Args:
        input: Node to apply abs().

    Returns:
        Node with positive valued features.
    """
    return AbsOperator(
        input=input,
    ).outputs["output"]
