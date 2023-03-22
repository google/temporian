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

"""DropIndex operator."""
from typing import List, Optional, Union

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class DropIndexOperator(Operator):
    """DropIndex operator."""

    def __init__(
        self,
        event: Event,
        labels: Optional[Union[List[str], str]] = None,
        keep: bool = True,
    ) -> None:
        super().__init__()

        # process labels
        labels = self._process_labels(event, labels)

        # input event
        self.add_input("event", event)

        # attributes
        self.add_attribute("labels", labels)
        self.add_attribute("keep", keep)

        # output features
        output_features = self._generate_output_features(event, labels, keep)

        # output sampling
        output_sampling = Sampling(
            index=[
                index_name
                for index_name in event.sampling().index()
                if index_name not in labels
            ]
        )

        # output event
        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )
        self.check()

    def _process_labels(
        self, event: Event, labels: Optional[Union[List[str], str]]
    ) -> List[str]:
        if labels is None:
            return event.sampling().index()

        if isinstance(labels, str):
            labels = [labels]

        missing_labels = [
            label for label in labels if label not in event.sampling().index()
        ]
        if missing_labels:
            raise KeyError(missing_labels)

        return labels

    def _generate_output_features(
        self, event: Event, labels: List[str], keep: bool
    ) -> List[Feature]:
        output_features = [
            Feature(name=feature.name, dtype=feature.dtype)
            for feature in event.features()
        ]
        if keep:
            output_features.extend(
                [
                    Feature(name=feature_name, dtype=None)  # TODO: fix dtype
                    for feature_name in labels
                ]
            )

        return output_features

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="DROP_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="labels",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
                    is_optional=False,
                ),
                pb.OperatorDef.Attribute(
                    key="keep",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(DropIndexOperator)


def drop_index(
    event: Event, labels: Optional[List[str]] = None, keep: bool = True
) -> Event:
    return DropIndexOperator(event, labels, keep).outputs()["event"]
