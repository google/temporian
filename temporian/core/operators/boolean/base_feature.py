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

"""Base boolean feature operator."""

from abc import ABC, abstractmethod

from temporian.core.data import dtype as dtype_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseBooleanFeatureOperator(Operator, ABC):
    """
    Base boolean feature operator to implement common logic.
    """

    def __init__(self, event_1: Event, event_2: Event):
        super().__init__()

        # inputs
        self.add_input("event_1", event_1)
        self.add_input("event_2", event_2)

        # ensure event_2 only has one feature
        if len(event_2.features) != 1:
            raise ValueError(
                "Event 2 must only have one feature. Current features: "
                f"{event_2.features}"
            )

        # ensure event_1 and event_2 have same sampling
        if event_1.sampling != event_2.sampling:
            raise ValueError(
                "Event 1 and event 2 must have same sampling. Current"
                f" samplings: {event_1.sampling}, {event_2.sampling}"
            )

        event_2_feature = event_2.features[0]

        # ensure all features in event_1 have same dtype as event_2's feature
        if not all(
            dtype_lib.same_kind(dtype, event_2.features[0].dtype)
            for dtype in event_1.dtypes.values()
        ):
            raise ValueError(
                "All features in event 1 must have the same dtype"
                f" ({event_2_feature.dtype}) as event 2's feature"
                f" {event_2_feature}. Current feature dtypes: {event_1.dtypes}"
            )

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=self.feature_name(f, event_2_feature),
                dtype=dtype_lib.BOOLEAN,
                sampling=f.sampling,
                creator=self,
            )
            for f in event_1.features
        ]

        output_sampling = event_1.sampling
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
            key=cls.operator_def_key,
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="event_1"),
                pb.OperatorDef.Input(key="event_2"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )

    @classmethod
    @property
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Get the key of the operator definition."""

    def feature_name(self, feature_1: Feature, feature_2: Feature) -> str:
        """Returns the name of the feature to be created."""
        return f"{feature_1.name}_{self.operation_name}_{feature_2.name}"

    @property
    @abstractmethod
    def operation_name(self) -> str:
        """Returns name of the operation to be performed."""
