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

"""Lag operator class and public API function definitions."""

from typing import List, Union

from temporian.core import operator_lib
from temporian.core.data.duration import Duration
from temporian.core.data.duration import duration_abbreviation
from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class LagOperator(Operator):
    def __init__(
        self,
        input: Node,
        duration: Duration,
    ):
        super().__init__()

        self._duration = duration
        self._duration_str = duration_abbreviation(duration)

        # inputs
        self.add_input("input", input)

        self.add_attribute("duration", duration)

        output_sampling = Sampling(
            index_levels=input.sampling.index.levels,
            creator=self,
            is_unix_timestamp=input.sampling.is_unix_timestamp,
        )

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f.name,
                dtype=f.dtype,
                sampling=output_sampling,
                creator=self,
            )
            for f in input.features
        ]

        self.add_output(
            "output",
            Node(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )
        self.check()

    @property
    def duration(self) -> Duration:
        return self._duration

    @property
    def duration_str(self) -> str:
        return self._duration_str

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="LAG",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="duration",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(LagOperator)


def _implementation(
    input: Node,
    duration: Union[Duration, List[Duration]],
    should_leak: bool = False,
) -> Node:
    """Lags or leaks `input` depending on `should_leak`."""

    if not isinstance(duration, list):
        duration = [duration]

    if not all(isinstance(d, (int, float)) and d > 0 for d in duration):
        raise ValueError(
            "duration must be a list of positive numbers. Got"
            f" {duration}, type {type(duration)}"
        )

    # Ensure that all durations are of type Duration. This converts ints to
    # float64 for consistent behavior.
    duration = [
        Duration(d) if not isinstance(d, Duration) else d for d in duration
    ]

    used_duration = duration if not should_leak else [-d for d in duration]

    if len(used_duration) == 1:
        return LagOperator(
            input=input,
            duration=used_duration[0],
        ).outputs["output"]

    return [
        LagOperator(input=input, duration=d).outputs["output"]
        for d in used_duration
    ]


def lag(
    input: Node, duration: Union[Duration, List[Duration]]
) -> Union[Node, List[Node]]:
    """Shifts the node's sampling forwards in time by a specified duration.

    Each timestamp in `input`'s sampling is shifted forwards by the specified
    duration. If `duration` is a list, then the input will be lagged by each
    duration in the list, and a list of nodes will be returned.

    Args:
        input: Node to lag the sampling of.
        duration: Duration or list of Durations to lag by.

    Returns:
        Lagged node, or list of lagged nodes if a Duration list was
        provided.
    """
    return _implementation(input=input, duration=duration)


def leak(
    input: Node, duration: Union[Duration, List[Duration]]
) -> Union[Node, List[Node]]:
    """Shifts the node's sampling backwards in time by a specified duration.

    Each timestamp in `input`'s sampling is shifted backwards by the specified
    duration. If `duration` is a list, then the input will be leaked by each
    duration in the list, and a list of nodes will be returned.

    Note that this operator moves future data into the past, and should be used
    with caution to prevent unwanted leakage.

    Args:
        input: Node to leak the sampling of.
        duration: Duration or list of Durations to leak by.

    Returns:
        Leaked node, or list of leaked nodes if a Duration list was
        provided.
    """
    return _implementation(input=input, duration=duration, should_leak=True)
