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

"""Simple moving average operator class and public API function definition.."""

from typing import Optional

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import EventSetNode
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.window.base import BaseWindowOperator
from temporian.utils.typecheck import typecheck
from temporian.core.typing import EventSetOrNode, WindowLength


class SimpleMovingAverageOperator(BaseWindowOperator):
    """
    Window operator to compute the simple moving average.
    """

    @classmethod
    def operator_def_key(cls) -> str:
        return "SIMPLE_MOVING_AVERAGE"

    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        if not feature.dtype.is_float:
            raise ValueError(
                "simple_moving_average requires floating point inputs."
                " Instead, get feature {feature.name!r} with type"
                f" {feature.dtype}. Note: You can cast features e.g."
                " `.cast(tp.float32)`"
            )
        return (
            DType.FLOAT32 if feature.dtype == DType.FLOAT32 else DType.FLOAT64
        )


operator_lib.register_operator(SimpleMovingAverageOperator)


@typecheck
@compile
def simple_moving_average(
    input: EventSetOrNode,
    window_length: WindowLength,
    sampling: Optional[EventSetOrNode] = None,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)
    if sampling is not None:
        assert isinstance(sampling, EventSetNode)

    return SimpleMovingAverageOperator(
        input=input,
        window_length=window_length,
        sampling=sampling,
    ).outputs["output"]
