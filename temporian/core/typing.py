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

from typing import Any, Callable, Dict, List, Set, Tuple, Type, TypeVar, Union
from temporian.core.data.dtype import DType
from temporian.core.data.duration import Duration

from temporian.core.data.node import EventSetNode
from temporian.core.event_set_ops import EventSetOperations
from temporian.core.types import MapExtras
from temporian.implementation.numpy.data.event_set import EventSet


EventSetNodeCollection = Union[
    EventSetNode, List[EventSetNode], Set[EventSetNode], Dict[str, EventSetNode]
]
"""A collection of [`EventSetNodes`][temporian.EventSetNode].

This can be a single EventSetNode, a list or set of EventSetNodes, or a
dictionary mapping names to EventSetNodes."""


# TODO: check why __doc__ (or help()) of EventSetOrNode shows TypeVar's doc
EventSetOrNode = TypeVar(
    "EventSetOrNode", EventSet, EventSetNode, EventSetOperations
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
own node using [`EventSet.node()`][temporian.EventSet.node], i.e., `[evset]`
is equivalent to `{evset.node() : evset}`.
"""

TypeOrDType = Union[DType, Type[float], Type[int], Type[str], Type[bool]]


IndexKeyItem = Union[int, str, bytes]
"""One of the values inside an [IndexKey][temporian.core.typing.IndexKey]."""


IndexKey = Union[Tuple[IndexKeyItem, ...], IndexKeyItem]
"""An index key is a tuple of values that identifies a single time sequence
inside an [`EventSet`][temporian.EventSet].

If, for example, your EventSet is indexed by `"name"` and `"number"`, with the
values on those being [`"Mark", "Sarah"]` and `[1, 2, 3]` respectively, the
possible index keys would be `("Mark", 1)`, `("Mark", 2)`, `("Mark", 3)`,
`("Sarah", 1)`, `("Sarah", 2)`, and `("Sarah", 3)`.

An index key can take the form of a single value (e.g. `"Mark"`) if it is being
used with an EventSet with a single index. In this case, using `"Mark"` is
equivalent to using `("Mark",)`.
"""

IndexKeyList = Union[IndexKey, List[IndexKey]]
"""A list of [`IndexKeys`][temporian.core.typing.IndexKey].

Auxiliary type to allow receiving a single IndexKey or a list of IndexKeys.

If receiving a single IndexKey, it is equivalent to receiving a list with a
single IndexKey.
"""

WindowLength = Union[Duration, EventSetOrNode]
"""Window length of a moving window operator.

A window length can be either constant or variable.

A constant window length is specified with a
[Duration][temporian.duration.Duration]. For example, `window_length=5.0` or
`window_length=tp.duration.days(4)`.

A variable window length is specified with an [EventSet][temporian.EventSet]
containing a single float64 feature. This EventSet can have the same sampling as
the input EventSet or a different one, in which case the output will have the
same sampling as the `window_length` EventSet. In both cases the feature value
on each timestamp will dictate the length of the window in that timestamp.

If an `EventSet`, it should contain strictly positive values. If receiving 0,
negative values, or missing values, the operator will treat the window as empty.
"""

Scalar = Union[int, float, str, bytes, bool]
"""A scalar value."""

MapFunction = Union[Callable[[int], Any], Callable[[int, MapExtras], Any]]
"""A function that maps an [`EventSet`][temporian.EventSet]'s value to another
value.

The function must receive the original value and optionally a
[`MapExtras`][temporian.types.MapExtras] object, which includes additional
information about the value's position in the EventSet, and return the new
value.
"""


# Internal

# Internally we only allow normalized index keys, i.e., tuples of bytes and ints
# IndexKeyItem and IndexKey are user-facing and are normalized before being
# passed to implementations/serialized/etc.
NormalizedIndexKeyItem = Union[int, bytes]
NormalizedIndexKey = Tuple[NormalizedIndexKeyItem, ...]
