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

        # TODO: Constraint the type of features supported in "add_event".

        if event.sampling() != add_index.sampling():
            raise ValueError(
                "event and add_index should have the same sampling"
            )

        if len(add_index.features()) == 0:
            raise ValueError("add_index contains no features")

        self._added_index = [k.name() for k in add_index.features()]

        overlap_features = set(self._added_index).intersection(
            event.sampling().index()
        )
        if len(overlap_features) > 0:
            raise ValueError(
                "add_index contains feature names already present in the"
                f" index: {list(overlap_features)}"
            )

        new_index = event.sampling().index() + self._added_index
        sampling = Sampling(index=new_index, creator=self)

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

    def added_index(self) -> list[str]:
        """New items in the index."""

        return self._added_index

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
    add_index: Union[Event, str, List[str]],
) -> Event:
    """Extands index and propagates feature values.

    Extends the index of `event` over the features of `add_event`. Feature
    values from `event` are duplicated over the new index. `add_event` can be a
    string or list of string representing features in `event`, or an event.

    For example, suppose an index-less event sequence containing two features
    "f_1" and "f_2", and containing 16 timestamps. Suppose "f_1" is a numerical
    feature and "f_2" is a string feature with 4 unique values across the 16
    timestamps. "Propagating" feature "f_1" over "f_2" will create an event
    sequence containing feature "f_1" and indexed by feature "f_2". This event
    sequence will contain 4 indexed time sequences each containing 16
    timestamps.

    Example:

    If `event` is indexed by ["x"] and contain two features "f1" and "f2", and
    if `add_event` is also indexed by ["x"] and contains two features "y" and
    "z", the result is an event indexed by ["x", "y", "z"] and containing
    features "f1" and "f2".

    Constraints:

    - If `add_index` is an event, `event` and `add_index` have the same index
      and same sampling.

    Args:
        event: The event to propagate.
        add_index: The features to index over. If add_index` is a list of
          strings, those string should refer to existing features of `event`

    Returns:
        An event sequence propagated over `add_index`.
    """

    if isinstance(add_index, str):
        add_index = [add_index]

    if isinstance(add_index, list):
        add_index_set = set(add_index)
        add_index = select(event, add_index)
        remaining_features = [
            f.name() for f in event.features() if f.name not in add_index_set
        ]
        event = select(event, remaining_features)

    return Propagate(
        event=event,
        add_index=add_index,
    ).outputs()["event"]
