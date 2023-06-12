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

"""Base calendar operator class definition."""

from abc import ABC, abstractmethod
from typing import Optional


from temporian.core.data.duration_utils import NormalizedDuration
from temporian.core.data.dtypes.dtype import DType
from temporian.core.data.node import (
    Node,
    create_node_new_features_existing_sampling,
)
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseWindowOperator(Operator, ABC):
    """Interface definition and common logic for window operators."""

    def __init__(
        self,
        input: Node,
        window_length: NormalizedDuration,
        sampling: Optional[Node] = None,
    ):
        super().__init__()

        self._window_length = window_length
        self.add_attribute("window_length", window_length)

        self._has_sampling = sampling is not None
        if self._has_sampling:
            assert sampling is not None

            self.add_input("sampling", sampling)
            effective_sampling_node = sampling

            input.schema.check_compatible_index(sampling.schema)

        else:
            effective_sampling_node = input

        self.add_input("input", input)

        feature_schemas = [  # pylint: disable=g-complex-comprehension
            FeatureSchema(
                name=f.name,
                dtype=self.get_feature_dtype(f),
            )
            for f in input.schema.features
        ]

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=feature_schemas,
                sampling_node=effective_sampling_node,
                creator=self,
            ),
        )

        self.check()

    @property
    def window_length(self) -> NormalizedDuration:
        return self._window_length

    @property
    def has_sampling(self) -> bool:
        return self._has_sampling

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key=cls.operator_def_key(),
            attributes=[
                pb.OperatorDef.Attribute(
                    key="window_length",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="sampling", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @classmethod
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Gets the key of the operator definition."""

    @abstractmethod
    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        """Gets the dtype of the output feature."""
