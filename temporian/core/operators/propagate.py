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

"""Propagate operator."""

from typing import Optional, List, Union

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.data.duration import Duration
from temporian.core.operators.select import select
from temporian.core.data.sampling import Sampling


class Propagate(Operator):
    """Propagate operator."""

    def __init__(
        self,
        event: Event,
        add_index: Event,
    ):
        super().__init__()

        self.add_input("event", event)
        self.add_input("add_index", add_index)

        # TODO: Check for compatible feature type
        new_index = event.sampling().index() + [
            f.name for f in add_index.features()
        ]
        sampling = Sampling(index=new_index, creator=None)

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f.name(),
                dtype=f.dtype(),
                sampling=sampling,
                creator=self,
            )
            for f in event.features()
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
            key="PROPAGATE",
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="event"),
                pb.OperatorDef.Input(key="add_index"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(Propagate)


def propagate(
    event: Event,
    add_index: Union[Event, List[str]],
) -> Event:
    """Propagates / duplicate events to a new index or set of indexes.

    For example, suppose an index-less event sequence containing two features
    "f_1" and "f_2", and containing 16 timestamps. Suppose "f_1" is a numerical
    feature and "f_2" is a string feature with 4 unique values across the 16
    timestamps. "Propagating" feature "f_1" over "f_2" will create an event
    sequence containing feature "f_1" and indexed by feature "f_2". This event
    sequence will contain 4 indexed time sequences each containing 16
    timestamps.

    TODO: Need to standardize the naming convention.

    Args:
        event: The event sequence to propagate.
        add_index: The features to index over. If add_index` is a list of
          strings, those string should refer to existing features of `event`

    Returns:
        An event sequence propagated over `add_index`.
    """

    if isinstance(add_index, list):
        add_index = select(event=event, feature_names=add_index)
        # TODO: Remove "add_index" from "event".

    return Propagate(
        event=event,
        add_index=add_index,
    ).outputs()["event"]
