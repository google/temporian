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

"""Base binary operators class definition."""

from abc import abstractmethod

from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseBinaryOperator(Operator):
    """Base for common code of binary operators (between two events)."""

    def __init__(
        self,
        input_1: Node,
        input_2: Node,
    ):
        super().__init__()

        # inputs
        self.add_input("input_1", input_1)
        self.add_input("input_2", input_2)

        input_1.check_same_sampling(input_2)

        if len(input_1.schema.features) != len(input_2.schema.features):
            raise ValueError(
                "The left and right arguments should have the same number of "
                f"features. Left features = {input_1.schema.features}, right "
                f"features = {input_2.schema.features}. Note: The name of the "
                "features do not have to match: Features are combined "
                "index-wise."
            )

        # check that features have same dtype
        for feature_idx, (feature_1, feature_2) in enumerate(
            zip(input_1.schema.features, input_2.schema.features)
        ):
            if feature_1.dtype != feature_2.dtype:
                raise ValueError(
                    "The operator is applied feature-wise, and "
                    "corresponding features (with the same index) should have "
                    "the same dtype. However the dtypes of the "
                    f" {feature_idx}-th features don't match. Left argument ="
                    f" {feature_1}, right argument = {feature_2}."
                )

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            FeatureSchema(
                name=self.output_feature_name(feature_1, feature_2),
                dtype=self.output_feature_dtype(feature_1, feature_2),
            )
            for feature_1, feature_2 in zip(
                input_1.schema.features, input_2.schema.features
            )
        ]

        self.add_output(
            "output",
            Node.create_new_features_existing_sampling(
                features=output_features,
                sampling_node=input_1,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key=cls.operator_def_key,
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="input_1"),
                pb.OperatorDef.Input(key="input_2"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @classmethod
    @property
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Gets the key of the operator definition."""

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Gets the prefix to use for the output features."""

    def output_feature_name(
        self, feature_1: FeatureSchema, feature_2: FeatureSchema
    ) -> str:
        return f"{self.prefix}_{feature_1.name}_{feature_2.name}"

    def output_feature_dtype(
        self, feature_1: FeatureSchema, feature_2: FeatureSchema
    ) -> DType:
        del feature_2
        return feature_1.dtype
