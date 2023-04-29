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

"""Propagate operator class and public API function definition."""


from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class Propagate(Operator):
    def __init__(
        self,
        event: Event,
        sampling: Event,
    ):
        super().__init__()

        self.add_input("event", event)
        self.add_input("sampling", sampling)

        self._index_mapping: list[int] = []
        sampling_index_name = sampling.sampling.index.names
        sampling_index_dtypes = sampling.sampling.index.dtypes
        for index in event.sampling.index:
            try:
                sampling_idx = sampling_index_name.index(index.name)
                self._index_mapping.append(sampling_idx)
            except ValueError:
                raise ValueError(
                    "The index of event should be contained in the index of"
                    f' sampling. Index "{index.name}" from event is not'
                    " available in sampling. event.index:"
                    f" {event.sampling.index},"
                    f" sampling.index={sampling.sampling.index}."
                )
            if sampling_index_dtypes[sampling_idx] != index.dtype:
                raise ValueError(
                    f'The index "{index.name}" is found both in the event and'
                    " sampling argument. However, the dtype is different."
                    f" {index.dtype} != {sampling_index_dtypes[sampling_idx]}"
                )

        output_sampling = Sampling(
            index_levels=sampling.sampling.index, creator=self
        )

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f.name,
                dtype=f.dtype,
                sampling=output_sampling,
                creator=self,
            )
            for f in event.features
        ]

        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )

        self.check()

    @property
    def index_mapping(self):
        return self._index_mapping

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="PROPAGATE",
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="event"),
                pb.OperatorDef.Input(key="sampling"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(Propagate)


def propagate(
    event: Event,
    sampling: Event,
) -> Event:
    """Propagates feature values over a larger index.

    Given `event` and `sampling` where `event` contains a super index of
    `sampling` (e.g., the index of `event` is `["x"]`, and the index of sampling
    is `["x","y"]`), duplicates the features of "events" over the index of
    `sampling`.

    Example:

        Inputs:
            event:
                feature_1: ...
                feature_2: ...
                index: ["x"]
            sampling:
                index: ["x", "y"]

        Output:
            feature_1: ...
            feature_2: ...
            index: ["x", "y"]

    Args:
        event: Event to propagate.
        sampling: Index to propagate over.

    Returns:
        Event propagated over `sampling`'s index.
    """

    return Propagate(
        event=event,
        sampling=sampling,
    ).outputs["event"]
