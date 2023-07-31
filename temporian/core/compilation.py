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
from copy import copy
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Callable,
    TypeVar,
)
from temporian.core.data.node import EventSetNode
from temporian.implementation.numpy.data.event_set import EventSet


F = TypeVar("F", bound=Callable)


def compile(fn: Optional[F] = None, *, verbose: int = 0) -> F:
    """Compiles a Temporian function.

    A Temporian function is a function that takes
    [`EventSetOrNodes`][temporian.EventSetOrNode] as arguments and returns
    [`EventSetOrNodes`][temporian.EventSetOrNode] as outputs.

    Compiling a function allows Temporian to optimize the underlying graph
    defined by the operators used inside the function, making it run on
    [`EventSets`][temporian.EventSet] more efficiently than if it weren't
    compiled, both in terms of memory and speed.

    Compiling a function is a necessary step before saving it to a file with
    [`tp.save()`][temporian.save].

    The output can be a single EventSetOrNode, a list of EventSetOrNodes, or a
    dictionary of names to EventSetOrNodes.

    Example usage:
    ```python
    >>> @tp.compile
    ... def f(x: EventSetNode, y: EventSetNode) -> EventSetNode:
    ...     return x.prefix("pre_").cumsum() + y

    >>> evset = tp.event_set(
    ...     timestamps=[1, 2, 3],
    ...     features={"value": [10, 20, 30]},
    ... )

    >>> result = f(evset, evset)
    >>> isinstance(result, tp.EventSet)
    True

    ```

    Example usage with arguments:
    ```python
    >>> @tp.compile(verbose=1)
    ... def f(x: EventSetNode) -> EventSetNode:
    ...     return x.prefix("pre_")

    ```

    Args:
        fn: The function to compile. The function must take EventSetNodes as
            arguments (and may have other arguments of arbitrary types) and
            return EventSetNodes as outputs.
        verbose: If >0, prints details about the execution on the standard error
            output when the wrapped function is applied eagerly on EventSets.
            The larger the number, the more information is displayed.

    Returns:
        The compiled function.
    """

    def _compile(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            is_eager = None
            args = list(args)

            # EventSetNode -> EventSet mapping for eager evaluation
            inputs_map = {}

            for i, arg in enumerate(args):
                args[i], is_eager = _process_argument(
                    arg, is_eager=is_eager, inputs_map=inputs_map
                )

            for k, arg in kwargs.items():
                kwargs[k], is_eager = _process_argument(
                    arg, is_eager=is_eager, inputs_map=inputs_map
                )

            outputs = fn(*args, **kwargs)

            if is_eager is None:
                raise ValueError(
                    "Cannot compile a function without EventSet or EventSetNode"
                    " argument."
                )
            elif is_eager:
                from temporian.core.evaluation import run

                return run(query=outputs, input=inputs_map, verbose=verbose)

            return outputs

        wrapper.is_tp_compiled = True  # type: ignore

        return wrapper

    # Function is being called as a decorator
    if fn is not None:
        return _compile(fn)

    # Else the function is being called as a function, so we return a decorator
    # that will receive the function to compile.
    return _compile  # type: ignore


def _process_argument(
    obj: Any,
    is_eager: Optional[bool],
    inputs_map: Dict[EventSetNode, EventSet],
) -> Tuple[Any, Optional[bool]]:
    """Processes arguments to an operator by checking if its being used in eager
    mode and converting EventSets to EventSetNodes if so.

    Also checks that all arguments are of the same type (EventSet or EventSetNode), by
    checking that is_eager is consistent with the type of obj, and raising if
    not.

    Note that the inputs_map is modified in-place by this function.
    """
    if isinstance(obj, tuple):
        obj = list(obj)
        for i, v in enumerate(obj):
            obj[i], is_eager = _process_argument(v, is_eager, inputs_map)
        return tuple(obj), is_eager

    if isinstance(obj, list):
        obj = copy(obj)
        for i, v in enumerate(obj):
            obj[i], is_eager = _process_argument(v, is_eager, inputs_map)
        return obj, is_eager

    if isinstance(obj, dict):
        obj = copy(obj)
        for k, v in obj.items():
            obj[k], is_eager = _process_argument(v, is_eager, inputs_map)
        return obj, is_eager

    err = (
        "Cannot mix EventSets and EventSetNodes as inputs to an operator."
        " Either get the node corresponding to each EventSet with .node(), or"
        " pass EventSets only."
    )

    if isinstance(obj, EventSet):
        if is_eager is None:
            is_eager = True
        elif not is_eager:
            # If an EventSetNode had been received and we receive an EventSet, raise
            raise ValueError(err)
        node = obj.node()
        # Its fine to overwrite the same node since the corresponding EventSet
        # is guaranteed to be the same one.
        inputs_map[node] = obj
        obj = node

    elif isinstance(obj, EventSetNode):
        if is_eager is None:
            is_eager = False
        elif is_eager:
            # If an EventSet had been received and we receive an EventSetNode, raise
            raise ValueError(err)

    return obj, is_eager
