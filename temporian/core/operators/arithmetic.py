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

"""Arithmetic operator class and public API function definition."""

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
    Sums two events.

    If resolution is Resolution.PER_FEATURE_IDX each feature in `event_1` will
    be summed to the feature in `event_2` in its same index.
    If resolution is Resolution.PER_FEATURE_NAME each feature in `event_1` will
    be summed to the feature in `event_2` with its same name.

    Args:
        event_1: First event.
        event_2: Second event.
        resolution: Resolution strategy to use.

    Returns:
        Sum of the two events.
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
    Substracts two events.

    If resolution is Resolution.PER_FEATURE_IDX each feature in `event_2` will
    be subtracted from the feature in `event_1` in its same index.
    If resolution is Resolution.PER_FEATURE_NAME each feature in `event_2` will
    be subtracted from the feature in `event_1` with its same name.

    Args:
        event_1: First event.
        event_2: Second event.
        resolution: Resolution strategy to use.

    Returns:
        Substraction of the two events.
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
    Multiplies two events.

    If resolution is Resolution.PER_FEATURE_IDX each feature in `event_1` will
    be multiplied with the feature in `event_2` in its same index.
    If resolution is Resolution.PER_FEATURE_NAME each feature in `event_1` will
    be multiplied with the feature in `event_2` with its same name.

    Args:
        event_1: First event
        event_2: Second event
        resolution: Resolution strategy to use.

    Returns:
       Multiplication of the two events.
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
    Divides two events.

    If resolution is Resolution.PER_FEATURE_IDX each feature in `event_1` will
    be divided by the feature in its same index in `event_2`.
    If resolution is Resolution.PER_FEATURE_NAME each feature in `event_1` will
    be divided by feature in `event_2` with its same name.

    Args:
        numerator: Numerator event.
        denominator: Denominator event.
        resolution: Resolution strategy to use.

    Returns:
        Division of the two events.
    """
    return ArithmeticOperator(
        event_1=numerator,
        event_2=denominator,
        operation=ArithmeticOperation.DIVISION,
        resolution=resolution,
    ).outputs()["event"]
