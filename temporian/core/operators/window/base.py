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

"""Base calendar operator."""

from abc import ABC, abstractmethod
from typing import Optional


from temporian.core.data.duration import Duration
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseWindowOperator(Operator, ABC):
    """
    Base window operator to implement common logic.
    """

    def __init__(
        self,
        event: Event,
        window_length: Duration,
        sampling: Optional[Event] = None,
    ):
        super().__init__()

        self._window_length = window_length
        self.add_attribute("window_length", window_length)

        if sampling is not None:
            self.add_input("sampling", sampling)
            effective_sampling = sampling.sampling()
        else:
            effective_sampling = event.sampling()

        self.add_input("event", event)

        # output
        self.add_output(
            "event",
            Event(
                features=self.output_features(
                    event=event,
                    sampling=effective_sampling,
                ),
                sampling=effective_sampling,
                creator=self,
            ),
        )

        self.check()

    def window_length(self):
        return self._window_length

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key=cls.operator_def_key,
            attributes=[
                pb.OperatorDef.Attribute(
                    key="window_length",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
                pb.OperatorDef.Input(key="sampling", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )

    @classmethod
    @property
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Get the key of the operator definition."""

    @abstractmethod
    def output_features(
        self, event: Event, sampling: Sampling
    ) -> list[Feature]:
        """Get the list of features in the output event."""

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Get the prefix to use for the output features."""
