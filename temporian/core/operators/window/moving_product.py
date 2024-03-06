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

"""Moving Product operator class and public API function definition.."""

from typing import Optional

import numpy as np

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import EventSetNode
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.window.base import BaseWindowOperator
from temporian.utils.typecheck import typecheck
from temporian.core.typing import EventSetOrNode, WindowLength


class MovingProductOperator(BaseWindowOperator):
    """
    Window operator to compute the moving product.
    """

    @classmethod
    def operator_def_key(cls) -> str:
        return "MOVING_PRODUCT"

    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        if not feature.dtype.is_float:
            raise ValueError(
                "moving_product requires the input EventSet to contain"
                " floating point features only, but received feature"
                f" {feature.name!r} with type {feature.dtype}. Note: You can"
                " cast features e.g. `.cast(tp.float32)`"
            )
        return (
            DType.FLOAT32 if feature.dtype == DType.FLOAT32 else DType.FLOAT64
        )


operator_lib.register_operator(MovingProductOperator)


@typecheck
@compile
def moving_product(
    input: EventSetOrNode,
    window_length: WindowLength,
    sampling: Optional[EventSetOrNode] = None,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)
    if sampling is not None:
        assert isinstance(sampling, EventSetNode)

    return MovingProductOperator(
        input=input,
        window_length=window_length,
        sampling=sampling,
    ).outputs["output"]


@compile
def cumprod(
    input: EventSetOrNode,
    sampling: Optional[EventSetOrNode] = None,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)
    if sampling is not None:
        assert isinstance(sampling, EventSetNode)

    return MovingProductOperator(
        input=input, window_length=np.inf, sampling=sampling
    ).outputs["output"]
