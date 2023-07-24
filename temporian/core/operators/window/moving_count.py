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
from temporian.core.data.duration_utils import Duration, normalize_duration
from temporian.core.data.node import EventSetNode
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.window.base import BaseWindowOperator
from temporian.core.typing import EventSetOrNode


class MovingCountOperator(BaseWindowOperator):
    @classmethod
    def operator_def_key(cls) -> str:
        return "MOVING_COUNT"

    def get_feature_dtype(self, feature: FeatureSchema) -> DType:
        return DType.INT32


operator_lib.register_operator(MovingCountOperator)


@compile
def moving_count(
    input: EventSetOrNode,
    window_length: Duration,
    sampling: Optional[EventSetOrNode] = None,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)
    if sampling is not None:
        assert isinstance(sampling, EventSetNode)

    return MovingCountOperator(
        input=input,
        window_length=normalize_duration(window_length),
        sampling=sampling,
    ).outputs["output"]
