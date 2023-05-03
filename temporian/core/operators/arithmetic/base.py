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

"""Base arithmetic operator class definition."""

from abc import abstractmethod

from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseArithmeticOperator(Operator):
    """Interface definition and common logic for arithmetic operators."""

    def __init__(
        self,
        input_1: Node,
        input_2: Node,
    ):
        super().__init__()

        # inputs
        self.add_input("input_1", input_1)
        self.add_input("input_2", input_2)

        if input_1.sampling is not input_2.sampling:
            raise ValueError("input_1 and input_2 must have same sampling.")

        if len(input_1.features) != len(input_2.features):
            raise ValueError(
                "input_1 and input_2 must have same number of features."
            )

        # check that features have same dtype
        for feature_1, feature_2 in zip(input_1.features, input_2.features):
            if feature_1.dtype != feature_2.dtype:
                raise ValueError(
                    (
                        "input_1 and input_2 must have same dtype for each"
                        " feature."
                    ),
                    (
                        f"feature_1: {feature_1}, feature_2: {feature_2} have"
                        " dtypes:"
                    ),
                    f"{feature_1.dtype}, {feature_2.dtype}.",
                )

        sampling = input_1.sampling

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=self.output_feature_name(feature_1, feature_2),
                dtype=self.output_feature_dtype(feature_1, feature_2),
                sampling=sampling,
                creator=self,
            )
            for feature_1, feature_2 in zip(input_1.features, input_2.features)
        ]

        self.add_output(
            "output",
            Node(
                features=output_features,
                sampling=sampling,
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
        self, feature_1: Feature, feature_2: Feature
    ) -> str:
        return f"{self.prefix}_{feature_1.name}_{feature_2.name}"

    def output_feature_dtype(
        self, feature_1: Feature, feature_2: Feature
    ) -> DType:
        return feature_1.dtype
