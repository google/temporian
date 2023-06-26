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

from functools import wraps
from typing import Any, Optional, Tuple
from temporian.core.data.node import Node
from temporian.implementation.numpy.data.event_set import EventSet


def operator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        is_eager = None
        args = list(args)

        # Node -> EventSet mapping for eager evaluation
        inputs = {}

        for i, arg in enumerate(args):
            node, is_eager = _process_operator_argument(arg, is_eager)
            if node is not None:
                args[i] = node
                inputs[node] = arg

        for k, arg in kwargs.items():
            node, is_eager = _process_operator_argument(arg, is_eager)
            if node is not None:
                kwargs[k] = node
                inputs[node] = arg

        output = fn(*args, **kwargs)

        if is_eager:
            evset = output.evaluate(inputs)
            # Prevent .node() from creating a new sampling when called
            evset._internal_node = output  # pylint: disable=protected-access
            return evset

        return output

    return wrapper


def _process_operator_argument(
    obj: Any, is_eager: Optional[bool]
) -> Tuple[Optional[EventSet], bool]:
    """Processes arguments to an operator by checking if its being used in eager
    mode and converting EventSets to Nodes if so.

    Also checks that all arguments are of the same type (EventSet or Node), by
    checking that is_eager is consistent with the type of obj, and raising if
    not."""
    node = None
    err = (
        "Cannot mix EventSets and Nodes as inputs to an operator. Either get"
        " the node corresponding to each EventSet with .node(), or pass"
        " EventSets only."
    )

    if isinstance(obj, EventSet):
        if is_eager is None:
            is_eager = True
        elif not is_eager:
            # If a Node had been received and we receive an EventSet, raise
            raise ValueError(err)
        node = obj.node()

    elif isinstance(obj, Node):
        if is_eager is None:
            is_eager = False
        elif is_eager:
            # If an EventSet had been received and we receive a Node, raise
            raise ValueError(err)

    return node, is_eager
