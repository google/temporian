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

"""ReIndex operator."""
from typing import List

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class ReIndex(Operator):
    def __init__(self, event: Event, dst_index: str | List[str]):
        super().__init__()

        # store destination index
        if isinstance(dst_index, str):
            dst_index = [dst_index]
        self.add_attribute("dst_index", dst_index)

        # verify dst index names exist in event
        dst_index_set = set(dst_index)
        event_features_set = {feature.name() for feature in event.features()}
        event_features_set.update(event.sampling().index())
        if not set(dst_index_set).issubset(event_features_set):
            raise KeyError(dst_index_set.difference(event_features_set))

        # inputs
        self.add_input("event", event)

        # outputs
        output_features = [
            Feature(feature_name, creator=self)
            for feature_name in event.sampling().index()
            if feature_name not in dst_index
        ] + [
            feature for feature in event.features() if feature not in dst_index
        ]
        output_sampling = Sampling(index=dst_index, creator=self)
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
            key="REINDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="dst_index",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(ReIndex)


def reindex(
    event: Event,
    dst_index: str | List[str],  # TODO: improve API interface
) -> Event:
    return ReIndex(event, dst_index).outputs()["event"]
