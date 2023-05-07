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

"""Base calendar operator class definition."""

from abc import ABC, abstractmethod
from typing import Optional, List


from temporian.core.data.duration import Duration
from temporian.core.data import dtype
from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class BaseWindowOperator(Operator, ABC):
    """Interface definition and common logic for window operators."""

    def __init__(
        self,
        input: Node,
        window_length: Duration,
        sampling: Optional[Node] = None,
    ):
        super().__init__()

        self._window_length = window_length
        self.add_attribute("window_length", window_length)

        if sampling is not None:
            self.add_input("sampling", sampling)
            self._has_sampling = True
            effective_sampling = sampling.sampling

            if event.sampling.index != sampling.sampling.index:
                raise ValueError(
                    "Event and sampling do not have the same index."
                    f" {event.sampling.index} != {sampling.sampling.index}"
                )

        else:
            effective_sampling = input.sampling
            self._has_sampling = False

        self.add_input("input", input)

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f.name,
                dtype=self.get_feature_dtype(f),
                sampling=effective_sampling,
                creator=self,
            )
            for f in input.features
        ]
        self._output_dtypes = [feature.dtype for feature in output_features]

        # output
        self.add_output(
            "output",
            Node(
                features=output_features,
                sampling=effective_sampling,
                creator=self,
            ),
        )

        self.check()

    @property
    def window_length(self) -> Duration:
        return self._window_length

    @property
    def has_sampling(self) -> bool:
        return self._has_sampling

    @property
    def output_dtypes(self) -> List[dtype.DType]:
        return self._output_dtypes

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
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="sampling", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @classmethod
    @property
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Gets the key of the operator definition."""

    @abstractmethod
    def get_feature_dtype(self, feature: Feature) -> dtype.DType:
        """Gets the dtype of the output feature."""
