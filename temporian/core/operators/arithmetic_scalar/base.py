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
from typing import Union
from abc import abstractmethod

from temporian.core.data.dtype import DType
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
        value: Union[float, int, str, bool],
    ):
        super().__init__()

        # inputs
        self.add_input("event", event)

        self.add_attribute("value", value)

        # check that every dtype of event feature is equal to value dtype
        value_dtype = python_type_to_temporian_dtype(type(value))

        # TODO: Check if we want to compare kind of dtype or just dtype
        # it makes sense to allow subtypes of the same kind. Value will
        # always be 64 bits. We need a way to allow 32 bits.
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
                dtype=self.output_feature_dtype(feature),
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

    def output_feature_dtype(self, feature: Feature) -> DType:
        return feature.dtype
