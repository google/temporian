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

"""Fillna operator."""

from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
)
from temporian.core.typing import EventSetOrNode
from temporian.utils.typecheck import typecheck
from temporian.core.operators.glue import glue
from temporian.core.data.dtype import DType


@typecheck
@compile
def fillna(
    input: EventSetOrNode,
    value: float,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)
    # TODO: Implement as a standalone operator.
    output = []
    for feature in input.schema.features:
        selected = input[feature.name]
        if feature.dtype in [DType.FLOAT32, DType.FLOAT64]:
            selected = selected.isnan().where(value, selected)
        output.append(selected)
    return glue(*output)
