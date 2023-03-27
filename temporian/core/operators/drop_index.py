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
        index_names: List[str],
        keep,
    ) -> None:
        super().__init__()

        # input event
        self.add_input("event", event)

        # attributes
        self.add_attribute("index_names", index_names)
        self.add_attribute("keep", keep)

        # output features
        output_features = self._generate_output_features(
            event, index_names, keep
        )

        # output sampling
        output_sampling = Sampling(
            index=[
                index_name
                for index_name in event.sampling().index()
                if index_name not in index_names
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

    def _generate_output_features(
        self, event: Event, index_names: List[str], keep: bool
    ) -> List[Feature]:
        output_features = [
            Feature(name=feature.name, dtype=feature.dtype)
            for feature in event.features()
        ]
        if keep:
            for feature_name in index_names:
                # check no other feature exists with this name
                if feature_name in event.feature_names:
                    raise ValueError(
                        f"Feature name {feature_name} coming from index already"
                        " exists in event."
                    )  # TODO: add automatic suffix instead of raising error? add capability to rename index

                output_features.append(
                    Feature(name=feature_name, dtype=None)  # TODO: fix dtype
                )

        return output_features

    @property
    def dst_feat_names(self) -> List[str]:
        feature_names = self.inputs()["event"].feature_names
        return (
            self.attributes()["index_names"] + feature_names
            if self.attributes()["keep"]
            else feature_names
        )

    @property
    def dst_index_names(self) -> List[str]:
        return [
            name
            for name in self.inputs()["event"].sampling().index()
            if name not in self.attributes()["index_names"]
        ]

    @property
    def index_names(self) -> List[str]:
        return self.attributes()["index_names"]

    @property
    def keep(self) -> bool:
        return self.attributes()["keep"]

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="DROP_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="index_names",
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


def _process_index_names(
    event: Event, index_names: Optional[Union[List[str], str]]
) -> List[str]:
    if index_names is None:
        return event.sampling().index()

    if isinstance(index_names, str):
        index_names = [index_names]

    if len(index_names) == 0:
        raise ValueError("Cannot specify empty list as `index_names` argument.")

    # check if any index names are missing from the index
    missing_index_names = [
        index_name
        for index_name in index_names
        if index_name not in event.sampling().index()
    ]
    if missing_index_names:
        raise KeyError(missing_index_names)

    return index_names


def drop_index(
    event: Event,
    index_names: Optional[Union[str, List[str]]] = None,
    keep: bool = True,
) -> Event:
    """_summary_

    Args:
        event: _description_
        index_names: _description_. Defaults to None.
        keep: _description_. Defaults to True.

    Returns:
        _description_
    """
    index_names = _process_index_names(event, index_names)
    return DropIndexOperator(event, index_names, keep).outputs()["event"]
