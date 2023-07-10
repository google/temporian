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

from dataclasses import dataclass, field
from typing import List, Set

from temporian.core.data.node import EventSetNode
from temporian.core.operators.base import Operator


@dataclass
class ScheduleStep:
    op: Operator

    # List of nodes that will not be used anymore after "op" is executed.
    released_nodes: List[EventSetNode]


@dataclass
class Schedule:
    steps: List[ScheduleStep] = field(default_factory=list)
    input_nodes: Set[EventSetNode] = field(default_factory=set)
