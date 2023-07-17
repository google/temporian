from typing import Dict, List, Set, TypeVar, Union

from temporian.core.data.node import EventSetNode
from temporian.implementation.numpy.data.event_set import EventSet


EventSetNodeCollection = Union[
    EventSetNode, List[EventSetNode], Set[EventSetNode], Dict[str, EventSetNode]
]
"""A collection of [`EventSetNodes`][temporian.EventSetNode].

This can be a single EventSetNode, a list or set of EventSetNodes, or a
dictionary mapping names to EventSetNodes."""


EventSetOrNode = TypeVar("EventSetOrNode", EventSet, EventSetNode)
"""Generic type for defining the input and output types of operators and
Temporian functions.

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

EventSetAndNode = Union[EventSet, EventSetNode]
"""An [`EventSet`][temporian.EventSet] or
[`EventSetNode`][temporian.EventSetNode]."""

EventSetAndNodeCollection = Union[
    EventSetAndNode,
    List[EventSetAndNode],
    Set[EventSetAndNode],
    Dict[str, EventSetAndNode],
]
"""A collection of [`EventSetAndNodes`][temporian.EventSetAndNode].

This can be a single EventSetAndNode, a list or set of EventSetAndNodes, or a
dictionary mapping names to EventSetAndNodes."""
