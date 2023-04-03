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

"""Base class for arithmetic operators"""

from abc import ABC, abstractmethod
from enum import Enum

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class Resolution(str, Enum):
    PER_FEATURE_IDX = "PER_FEATURE_IDX"
    PER_FEATURE_NAME = "PER_FEATURE_NAME"


class BaseArithmeticOperator(Operator, ABC):
    """Base Arithmetic operator."""

    def __init__(
        self,
        event_1: Event,
        event_2: Event,
        resolution: Resolution,
    ):
        super().__init__()

        # inputs
        self.add_input("event_1", event_1)
        self.add_input("event_2", event_2)

        self.add_attribute("resolution", resolution)

        if not isinstance(resolution, Resolution):
            raise ValueError("resolution must be a Resolution.")

        if event_1.sampling() is not event_2.sampling():
            raise ValueError("event_1 and event_2 must have same sampling.")

        if len(event_1.features()) != len(event_2.features()):
            raise ValueError(
                "event_1 and event_2 must have same number of features."
            )

        # check that features have same dtype
        for feature_1, feature_2 in zip(event_1.features(), event_2.features()):
            if feature_1.dtype() != feature_2.dtype():
                raise ValueError(
                    (
                        "event_1 and event_2 must have same dtype for each"
                        " feature."
                    ),
                    (
                        f"feature_1: {feature_1}, feature_2: {feature_2} have"
                        " dtypes:"
                    ),
                    f"{feature_1.dtype()}, {feature_2.dtype()}.",
                )

        sampling = event_1.sampling()

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"{self.prefix}_{feature_1.name()}_{feature_2.name()}",
                dtype=feature_1.dtype(),
                sampling=sampling,
                creator=self,
            )
            for feature_1, feature_2 in zip(
                event_1.features(), event_2.features()
            )
        ]

        self.add_output(
            "event",
            Event(
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
            attributes=[
                pb.OperatorDef.Attribute(
                    key="resolution",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event_1"),
                pb.OperatorDef.Input(key="event_2"),
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
