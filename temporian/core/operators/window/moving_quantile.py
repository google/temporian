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

"""Moving count operator class and public API function definition."""

from typing import Optional

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import EventSetNode
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.window.base import BaseWindowOperator
from temporian.core.typing import EventSetOrNode, WindowLength
from temporian.proto import core_pb2 as pb


class MovingQuantileOperator(BaseWindowOperator):
    extra_attribute_def = [
        {
            "key": "quantile",
            "is_optional": True,
            "type": pb.OperatorDef.Attribute.Type.FLOAT_64,
        }
    ]

    def __init__(
        self,
        input: EventSetNode,
        window_length: WindowLength,
        quantile: float,
        sampling: Optional[EventSetNode] = None,
    ):
        if quantile < 0 or quantile > 1:
            raise ValueError(
                (
                    "`quantile` must be a float between 0 and 1. "
                    f"Received {quantile}"
                )
            )
        self.quantile = quantile
        super().__init__(input, window_length, sampling)

    def add_extra_attributes(self):
        self.add_attribute("quantile", self.quantile)

    @classmethod
    def operator_def_key(cls) -> str:
        return "MOVING_QUANTILE"

    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        if not feature.dtype.is_numerical:
            raise ValueError(
                "moving_quantile requires the input EventSet to contain numerical"
                f" features only, but received feature {feature.name!r} with"
                f" type {feature.dtype}"
            )
        if feature.dtype.is_integer:
            return DType.FLOAT32
        return feature.dtype


operator_lib.register_operator(MovingQuantileOperator)


@compile
def moving_quantile(
    input: EventSetOrNode,
    window_length: WindowLength,
    quantile: float,
    sampling: Optional[EventSetOrNode] = None,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)
    if sampling is not None:
        assert isinstance(sampling, EventSetNode)

    return MovingQuantileOperator(
        input=input,
        window_length=window_length,
        quantile=quantile,
        sampling=sampling,
    ).outputs["output"]
