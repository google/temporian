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

"""Filter operator."""
from temporian.core import operator_lib
from temporian.core.data.dtype import BOOLEAN
from temporian.core.data.feature import Feature
from temporian.core.data.event import Event
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class FilterOperator(Operator):
    """Filter operator."""

    def __init__(self, event: Event, condition: Event):
        super().__init__()

        # check that condition is a single feature
        if len(condition.features) != 1:
            raise ValueError(
                "Condition must be a single feature. Got"
                f" {len(condition.features)} instead."
            )

        # check that condition is a boolean feature
        if condition.features[0].type != BOOLEAN:
            raise ValueError(
                "Condition must be a boolean feature. Got"
                f" {condition.features[0].type} instead."
            )

        # check both events have same sampling
        if event.sampling != condition.sampling:
            raise ValueError(
                "Event and condition must have the same sampling. Got"
                f" {event.sampling} and {condition.sampling} instead."
            )

        # inputs
        self.add_input("event", event)
        self.add_input("condition", condition)

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"{f.name}_filtered_by_{condition.features[0].name}",
                dtype=f.dtype,
                sampling=event.sampling,
                creator=self,
            )
            for f in event.features
        ]

        output_sampling = event.sampling
        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="FILTER",
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="event"),
                pb.OperatorDef.Input(key="condition"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(FilterOperator)


# Added `_event` to the function name to avoid name collision with the
# `filter` function in the `builtins` module.
def filter_event(
    event: Event,
    condition: Event,
) -> Event:
    """Filter operator.

    Removes event features samplings and values where the condition is False.

    Args:
        event: event to filter
        condition: event with a single boolean feature condition.

    Returns:
        Event: filtered event.
    """
    return FilterOperator(event, condition).outputs["event"]