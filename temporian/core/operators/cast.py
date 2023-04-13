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

"""Type cast operator."""
from typing import List, Union, Mapping
from temporian import Feature

from temporian.core.data.dtype import DType
from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class CastOperator(Operator):
    """Type cast operator."""

    def __init__(
        self,
        event: Event,
        to: Union(DType, Mapping[Union[str, DType], DType]),
    ):
        super().__init__()

        # Check that all origin keys are DType or feature_names
        if not isinstance(to, DType):
            for origin_key in to:
                if (
                    not isinstance(origin_key, DType)
                    and origin_key not in event.feature_names
                ):
                    raise ValueError(f"Invalid key to cast: {origin_key}")

        # Convert any input format to feature_name->target_dtype
        target_dtypes = {}
        for feature in event.features:
            if isinstance(to, DType):
                target_dtypes[feature.name] = to
            elif feature.name in to:
                target_dtypes[feature.name] = to[feature.name]
            elif feature.dtype in to:
                target_dtypes[feature.name] = to[feature.dtype]
            else:
                target_dtypes[feature.name] = feature.dtype
        self.add_attribute("target_dtypes", target_dtypes)

        # inputs
        self.add_input("event", event)

        # outputs
        output_features = []
        reuse_event = True
        for feature in event.features:
            if target_dtypes[feature.name] is feature.dtype:
                # Reuse feature
                output_features.append(feature)
            else:
                # Create new feature
                reuse_event = False
                output_features.append(
                    # Note: we're not renaming output features here
                    Feature(
                        feature.name,
                        target_dtypes[feature.name],
                        feature.sampling,
                        creator=self,
                    )
                )

        # Output event: don't create new if all features are reused
        self.add_output(
            "event",
            event
            if reuse_event
            else Event(
                features=output_features,
                sampling=event.sampling,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="CAST",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="target_dtypes",
                    type=pb.OperatorDef.Attribute.Type.MAP_STR_STR,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(CastOperator)


def cast(
    event: Event,
    to: Union(DType, Mapping[Union[str, DType], DType]),
) -> Event:
    return CastOperator(event, to).outputs["event"]
