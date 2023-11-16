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


"""Drop operator class and public API function definitions."""

from typing import List, Union
from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck
from temporian.core.operators.select import select


@typecheck
@compile
def drop(
    input: EventSetOrNode, feature_names: Union[str, List[str]]
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    if isinstance(feature_names, str):
        feature_names = [feature_names]

    input_features = input.schema.feature_names()

    if not all([fn in input_features for fn in feature_names]):
        raise TypeError(
            "Features"
            f" {[fn for fn in feature_names if fn not in input_features]} are"
            " not present in the input"
        )

    return select(
        input=input,
        feature_names=[fn for fn in input_features if fn not in feature_names],
    )
