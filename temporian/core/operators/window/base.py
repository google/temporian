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
from typing import Any, List, Optional, Mapping
from temporian.core.data.duration_utils import normalize_duration


from temporian.core.data.duration_utils import NormalizedDuration
from temporian.core.data.dtype import DType
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_existing_sampling,
)
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.base import Operator
from temporian.core.typing import WindowLength
from temporian.proto import core_pb2 as pb


class BaseWindowOperator(Operator, ABC):
    """Interface definition and common logic for window operators."""

    extra_attribute_def: List[Mapping[str, Any]] = []

    def __init__(
        self,
        input: EventSetNode,
        window_length: WindowLength,
        sampling: Optional[EventSetNode] = None,
    ):
        super().__init__()

        has_variable_winlen = isinstance(window_length, EventSetNode)
        self._has_variable_winlen = has_variable_winlen

        has_sampling = sampling is not None
        self._has_sampling = has_sampling

        if has_sampling:
            if has_variable_winlen:
                raise ValueError(
                    "`sampling` cannot be specified if a variable"
                    " `window_length` is specified with an EventSet. If"
                    " specifying `window_length` with an EventSet, that"
                    " EventSet's sampling will be used."
                )
            input.schema.check_compatible_index(sampling.schema)
            self.add_input("sampling", sampling)

        if has_variable_winlen:
            if (
                len(window_length.schema.features) != 1
                or window_length.schema.features[0].dtype != DType.FLOAT64
            ):
                raise ValueError(
                    "`window_length` must have exactly one float64 feature."
                )
            self.add_input("window_length", window_length)
            self._window_length = None
        else:
            window_length = normalize_duration(window_length)
            self.add_attribute("window_length", window_length)
            self._window_length = window_length

        self.add_input("input", input)

        # Note: effective_sampling_node can be either the received sampling,
        # window_length, or the input
        effective_sampling_node = (
            window_length
            if has_variable_winlen
            else (sampling if has_sampling else input)
        )
        assert isinstance(effective_sampling_node, EventSetNode)

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=self.feature_schema(input),
                sampling_node=effective_sampling_node,
                creator=self,
            ),
        )
        self.add_extra_attributes()

        self.check()

    def feature_schema(self, input: EventSetNode):
        return [  # pylint: disable=g-complex-comprehension
            FeatureSchema(
                name=f.name,
                dtype=self.get_feature_dtype(f),
            )
            for f in input.schema.features
        ]

    @property
    def window_length(self) -> Optional[NormalizedDuration]:
        """Returns None if window_length is variable (i.e. an EventSet was
        passed as `window_length` to the operator)."""
        return self._window_length

    @property
    def has_sampling(self) -> bool:
        return self._has_sampling

    @property
    def has_variable_winlen(self) -> bool:
        return self._has_variable_winlen

    def add_extra_attributes(self):
        pass

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        extra_attr_def = [
            pb.OperatorDef.Attribute(**attr) for attr in cls.extra_attribute_def
        ]
        return pb.OperatorDef(
            key=cls.operator_def_key(),
            attributes=[
                pb.OperatorDef.Attribute(
                    key="window_length",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=True,
                ),
                *extra_attr_def,
            ],
            inputs=[
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="sampling", is_optional=True),
                pb.OperatorDef.Input(key="window_length", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @classmethod
    @abstractmethod
    def operator_def_key(cls) -> str:
        """Gets the key of the operator definition."""

    @abstractmethod
    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        """Gets the dtype of the output feature."""
