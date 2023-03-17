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

"""Arithmetic operator."""

from enum import Enum

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class ArithmeticOperation(str, Enum):
    ADDITION = "ADDITION"
    SUBTRACTION = "SUBTRACTION"
    MULTIPLICATION = "MULTIPLICATION"
    DIVISION = "DIVISION"

    @staticmethod
    def prefix(operation: "ArithmeticOperation") -> str:
        if operation == ArithmeticOperation.ADDITION:
            return "add"
        if operation == ArithmeticOperation.SUBTRACTION:
            return "sub"
        if operation == ArithmeticOperation.MULTIPLICATION:
            return "mult"
        if operation == ArithmeticOperation.DIVISION:
            return "div"
        raise ValueError(f"Unknown operation: {operation}.")

    @staticmethod
    def is_valid(operation: str) -> bool:
        return operation in [op.value for op in ArithmeticOperation]


class Resolution(str, Enum):
    PER_FEATURE_IDX = "PER_FEATURE_IDX"
    PER_FEATURE_NAME = "PER_FEATURE_NAME"


class ArithmeticOperator(Operator):
    """Arithmetic operator."""

    def __init__(
        self,
        event_1: Event,
        event_2: Event,
        operation: ArithmeticOperation,
        resolution: Resolution,
    ):
        super().__init__()

        # inputs
        self.add_input("event_1", event_1)
        self.add_input("event_2", event_2)

        self.add_attribute("operation", operation)
        self.add_attribute("resolution", resolution)

        if not isinstance(operation, ArithmeticOperation):
            raise ValueError("operation must be an ArithmeticOperation.")

        if not isinstance(resolution, Resolution):
            raise ValueError("resolution must be a Resolution.")

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

        prefix = ArithmeticOperation.prefix(operation)

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"{prefix}_{feature_1.name()}_{feature_2.name()}",
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
            key="ARITHMETIC",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="operation",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                    is_optional=False,
                ),
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


operator_lib.register_operator(ArithmeticOperator)


def sum(
    event_1: Event,
    event_2: Event,
    resolution: Resolution = Resolution.PER_FEATURE_IDX,
) -> Event:
    """
    Sum two events.

    Args:
        event_1: First event
        event_2: Second event
        resolution: If resolution is Resolution.PER_FEATURE_IDX each feature
            will be sum index wise. If resolution is Resolution.PER_FEATURE_NAME
            each feature of event_1 will be sum with the feature in event_2
            with the same name. Defaults to Resolution.PER_FEATURE_IDX.


    Returns:
        Event: Sum of event_1 and event_2 according to resolution.
    """
    return ArithmeticOperator(
        event_1=event_1,
        event_2=event_2,
        operation=ArithmeticOperation.ADDITION,
        resolution=resolution,
    ).outputs()["event"]


def substract(
    event_1: Event,
    event_2: Event,
    resolution: Resolution = Resolution.PER_FEATURE_IDX,
) -> Event:
    """
    Substract two events.

    Args:
        event_1: First event
        event_2: Second event
        resolution: If resolution is Resolution.PER_FEATURE_IDX each feature
            will be substract index wise. If resolution is
            Resolution.PER_FEATURE_NAME each feature of event_1 will be
            substract with the feature in event_2 with the same name.
            Defaults to Resolution.PER_FEATURE_IDX.

    Returns:
        Event: Substraction of event_1 and event_2 according to resolution.
    """
    return ArithmeticOperator(
        event_1=event_1,
        event_2=event_2,
        operation=ArithmeticOperation.SUBTRACTION,
        resolution=resolution,
    ).outputs()["event"]


def multiply(
    event_1: Event,
    event_2: Event,
    resolution: Resolution = Resolution.PER_FEATURE_IDX,
) -> Event:
    """
    Multiply two events.

    Args:
        event_1: First event
        event_2: Second event
        resolution: If resolution is Resolution.PER_FEATURE_IDX each feature
            will be multiply index wise. If resolution is
            Resolution.PER_FEATURE_NAME each feature of event_1 will be
            multiply with the feature in event_2 with the same name.
            Defaults to Resolution.PER_FEATURE_IDX.

    Returns:
        Event: Multiplication of event_1 and event_2 according to resolution.
    """
    return ArithmeticOperator(
        event_1=event_1,
        event_2=event_2,
        operation=ArithmeticOperation.MULTIPLICATION,
        resolution=resolution,
    ).outputs()["event"]


def divide(
    numerator: Event,
    denominator: Event,
    resolution: Resolution = Resolution.PER_FEATURE_IDX,
) -> Event:
    """
    Divide two events.

    Args:
        numerator: Numerator event
        denominator: Denominator event
        resolution: If resolution is Resolution.PER_FEATURE_IDX each feature
            will be divide index wise. If resolution is
            Resolution.PER_FEATURE_NAME each feature of numerator will be
            divide with the feature in denominator with the same name.
            Defaults to Resolution.PER_FEATURE_IDX.

    Returns:
        Event: Division of numerator and denominator according to resolution.
    """
    return ArithmeticOperator(
        event_1=numerator,
        event_2=denominator,
        operation=ArithmeticOperation.DIVISION,
        resolution=resolution,
    ).outputs()["event"]
