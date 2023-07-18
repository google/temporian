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

from typing import Dict, List, Set, TypeVar, Union

from temporian.core.data.node import EventSetNode
from temporian.core.mixins import EventSetOperationsMixin
from temporian.implementation.numpy.data.event_set import EventSet


EventSetNodeCollection = Union[
    EventSetNode, List[EventSetNode], Set[EventSetNode], Dict[str, EventSetNode]
]
"""A collection of [`EventSetNodes`][temporian.EventSetNode].

This can be a single EventSetNode, a list or set of EventSetNodes, or a
dictionary mapping names to EventSetNodes."""


# TODO: check why __doc__ (or help()) of EventSetOrNode shows TypeVar's doc
EventSetOrNode = TypeVar(
    "EventSetOrNode", EventSet, EventSetNode, EventSetOperationsMixin
)
"""Generic type for an [`EventSet`][temporian.EventSet] or
[`EventSetNode`][temporian.EventSetNode].

Mainly used to define the input and output types of operators and Temporian
functions.

A function typed as `f(a: EventSetOrNode, ...) -> EventSetOrNode` indicates that
the function receives either EventSets or EventSetNodes as input, and returns
that same type as output. In other words, `f(evset)` returns an EventSet, and
`f(node)` returns an EventSetNode.
"""

EventSetCollection = Union[EventSet, List[EventSet], Dict[str, EventSet]]
"""A collection of [`EventSets`][temporian.EventSet].

This can be a single EventSet, a list of EventSets, or a dictionary mapping
names to EventSets."""

NodeToEventSetMapping = Union[
    Dict[EventSetNode, EventSet], EventSet, List[EventSet]
]
"""A mapping of [`EventSetNodes`][temporian.EventSetNode] to
[`EventSets`][temporian.EventSet].

If a dictionary, the mapping is defined by it.

If a single EventSet or a list of EventSets, each EventSet is mapped to their
own node using [`EventSet.node()`][temporian.EventSet.node], i.e., `[event_set]`
is equivalent to `{event_set.node() : event_set}`.
"""

EventSetAndNode = Union[EventSet, EventSetNode, EventSetOperationsMixin]
"""An [`EventSet`][temporian.EventSet] or
[`EventSetNode`][temporian.EventSetNode]."""

EventSetAndNodeCollection = Union[
    EventSetAndNode,
    List[EventSetAndNode],
    Set[EventSetAndNode],
    Dict[str, EventSetAndNode],
]
"""A collection of [`EventSetAndNodes`][temporian.core.typing.EventSetAndNode].

This can be a single EventSetAndNode, a list or set of EventSetAndNodes, or a
dictionary mapping names to EventSetAndNodes."""
