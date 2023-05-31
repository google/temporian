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

from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseCalendarOperator(Operator, ABC):
    """Interface definition and common logic for calendar operators."""

    def __init__(self, sampling: Node):
        super().__init__()

        if not sampling.schema.is_unix_timestamp:
            raise ValueError(
                "Calendar operators can only be applied on nodes with unix"
                " timestamps as sampling. This can be specified with"
                " `is_unix_timestamp=True` when manually creating a sampling."
            )

        # input
        self.add_input("sampling", sampling)

        self.add_output(
            "output",
            Node.create_new_features_existing_sampling(
                features=[(self.output_feature_name, DType.INT32)],
                sampling_node=sampling,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key=cls.operator_def_key,
            inputs=[pb.OperatorDef.Input(key="sampling")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @classmethod
    @property
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Gets the key of the operator definition."""

    @classmethod
    @property
    @abstractmethod
    def output_feature_name(cls) -> str:
        """Gets the name of the generated feature in the output node."""
