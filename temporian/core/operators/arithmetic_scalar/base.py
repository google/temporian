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

"""Base class for arithmetic scalar operators"""

from abc import abstractmethod

from temporian.core.data.dtype import python_type_to_temporian_dtype
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseArithmeticScalarOperator(Operator):
    """Base Arithmetic scalar operator."""

    def __init__(
        self,
        event: Event,
        value: any,
    ):
        super().__init__()

        # inputs
        self.add_input("event", event)

        # check that every dtype of event feature is equal to value dtype
        value_dtype = python_type_to_temporian_dtype(type(value))

        for feature in event.features:
            if feature.dtype != value_dtype:
                raise ValueError(
                    f"Feature {feature.name} has dtype {feature.dtype} "
                    f"but value has dtype {value_dtype}. Both must be equal."
                )

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"{self.prefix}_{feature.name}_{value}",
                dtype=value.dtype,
                sampling=event.sampling,
                creator=self,
            )
            for feature in event.features
        ]

        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=event.sampling,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key=cls.operator_def_key,
            attributes=[
                pb.OperatorDef.Attribute(
                    key="value",
                    type=pb.OperatorDef.Attribute.Type.ANY,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )

    @classmethod
    @property
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Get the key of the operator definition."""

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Get the prefix to use for the output features."""
