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

"""Construction and evaluation of an operator schedule for a set of inputs."""

import time
import sys
from typing import Dict, List, Set, Union, Optional
from collections import defaultdict

from temporian.core.data.node import Node
from temporian.core.operators.base import Operator
from temporian.implementation.numpy import evaluation as np_eval
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core.graph import infer_graph
from temporian.core.schedule import Schedule

EvaluationQuery = Union[Node, List[Node], Set[Node], Dict[str, Node]]
EvaluationInput = Union[
    # A dict of Nodes to corresponding EventSet.
    Dict[Node, EventSet],
    # A single EventSet. Equivalent to {event_set.node() : event_set}.
    EventSet,
    # A list of EventSets. Feed each EventSet individually like EventSet.
    List[EventSet],
]
EvaluationResult = Union[EventSet, List[EventSet], Dict[str, EventSet]]


def run(
    query: EvaluationQuery,
    input: EvaluationInput,
    verbose: int = 0,
    check_execution: bool = True,
) -> EvaluationResult:
    """Evaluates [`Nodes`][temporian.Node] on [`EventSets`][temporian.EventSet].

    Performs all computation defined by the graph between the `query` Nodes and
    the `input` EventSets.

    The result is returned in the same format as the `query` argument.

    Single input output example:
        ```python
        >>> input_evset = tp.event_set(timestamps=[1, 2, 3], features={"f": [0, 4, 10]})
        >>> input_node = input_evset.node()
        >>> output_node = tp.moving_sum(input_node, 5)
        >>> output_evset = tp.run(output_node, input_evset)

        >>> # Equivalent
        >>> output_evset = output_node.run(input_evset)

        >>> # Also equivalent
        >>> output_evset = tp.run(output_node, {input_node: input_evset})

        ```

    Multiple inputs and outputs example:
        ```python
        >>> evset_1 = tp.event_set(timestamps=[1, 2, 3], features={"f1": [0.1, 42, 10]})
        >>> evset_2 = tp.event_set(timestamps=[1, 2, 3],
        ...     features={"f2": [-1.5, 50, 30]},
        ...     same_sampling_as=evset_1
        ... )

        >>> # Graph with 2 inputs and 2 steps
        >>> input_1 = evset_1.node()
        >>> input_2 = evset_2.node()
        >>> step_1 = input_1 + input_2
        >>> step_2 = tp.simple_moving_average(step_1, 2)

        >>> # Get step_1 and step_2 at once
        >>> evset_step_1, evset_step_2 = tp.run([step_1, step_2],
        ...     {input_1: evset_1, input_2: evset_2}
        ... )

        >>> # Equivalent
        evset_step_1, evset_step_2 = tp.run(
        ...     [step_1, step_2],
        ...     [evset_1, evset_2],
        ... )

        >>> # Also equivalent. EventSets are mapped by their .node(), not by position.
        >>> evset_step_1, evset_step_2 = tp.run(
        ...     [step_1, step_2],
        ...     [evset_2, evset_1],
        ... )

        ```

    Args:
        query: Nodes to compute. Supports Node, dict of Nodes and list of Nodes.
        input: Event sets to be used for the computation. Supports EventSet,
            list of EventSets, dict of Nodes to EventSets, and dict of Node
            names to EventSet. If a single EventSet or list of EventSet,
            they must be named and will be used as input for the Nodes with the
            same name. If a dict of Node names to EventSet, they will be used
            as input for the Nodes with those names. If a dict of Nodes to event
            sets, they will be used as input for those Nodes.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.
        check_execution: If true, the input and output of the op implementation
            are validated to check any bug in the library internal code. If
            false, checks are skipped.

    Returns:
        An object with the same structure as `query` containing the results.
            If `query` is a dictionary of Nodes, the returned object will be a
            dictionary of EventSet. If `query` is a list of Nodes, the
            returned value will be a list of EventSet with the same order.
    """
    # TODO: Create an internal configuration object for options such as
    # `check_execution`.

    begin_time = time.perf_counter()

    input = _normalize_input(input)
    normalized_query = _normalize_query(query)

    if verbose >= 1:
        print("Build schedule", file=sys.stderr)

    # Schedule execution
    assert isinstance(normalized_query, set)
    input_nodes = set(input.keys())
    schedule = build_schedule(
        inputs=input_nodes, outputs=normalized_query, verbose=verbose
    )

    if verbose == 1:
        print(
            f"Run {len(schedule.ordered_operators)} operators",
            file=sys.stderr,
        )

    elif verbose >= 2:
        print("Schedule:\n", schedule, file=sys.stderr)

    # Evaluate schedule
    #
    # Note: "outputs" is a dictionary of event (including the query events) to
    # event data.
    outputs = np_eval.run_schedule(
        input,
        schedule,
        verbose=verbose,
        check_execution=check_execution,
    )

    end_time = time.perf_counter()

    if verbose == 1:
        print(f"Execution in {end_time - begin_time:.5f} s", file=sys.stderr)

    return _denormalize_outputs(outputs, query)


def build_schedule(
    inputs: Optional[Set[Node]], outputs: Set[Node], verbose: int = 0
) -> Schedule:
    """Calculates which operators need to be executed in which order to compute
    a set of output Nodes given a set of input Nodes.

    This implementation is based on Kahn's algorithm.

    Args:
        inputs: Input Nodes.
        outputs: Output Nodes.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.

    Returns:
        Tuple of:
            - Ordered list of operators, such that the first operator should be
            computed before the second, second before the third, etc.
            - Mapping of Node name inputs to Nodes. The keys are the string
            values in the `inputs` argument, and the values are the Nodes
            corresponding to each one. If a value was already a Node, it won't
            be present in the returned dictionary.
    """
    # List all Nodes and operators in between inputs and outputs.
    #
    # Fails if the outputs cannot be computed from the inputs e.g. some inputs
    # are missing.
    graph = infer_graph(inputs, outputs)

    schedule = Schedule(input_nodes=graph.inputs)

    if verbose >= 2:
        print("Graph:\n", graph, file=sys.stderr)

    # Operators ready to be computed (i.e. ready to be added to "planned_ops")
    # as all their inputs are already computed by "planned_ops" or specified by
    # "inputs".
    ready_ops: Set[Operator] = set()

    # "node_to_op[e]" is the list of operators with node "e" as input.
    node_to_op: Dict[Node, List[Operator]] = defaultdict(lambda: [])

    # "op_to_num_pending_inputs[op]" is the number of "not yet scheduled" inputs
    # of operator "op". Operators in "op_to_num_pending_inputs" have not yet
    # scheduled.
    op_to_num_pending_inputs: Dict[Operator, int] = defaultdict(lambda: 0)

    # Compute "node_to_op" and "op_to_num_pending_inputs".
    for op in graph.operators:
        num_pending_inputs = 0
        for input_node in op.inputs.values():
            if input_node in graph.inputs:
                # This input is already available
                continue
            node_to_op[input_node].append(op)
            num_pending_inputs += 1
        if num_pending_inputs == 0:
            # Ready to be scheduled
            ready_ops.add(op)
        else:
            # Some of the inputs are missing.
            op_to_num_pending_inputs[op] = num_pending_inputs

    # Compute the schedule
    while ready_ops:
        # Get an op ready to be scheduled
        op = ready_ops.pop()

        # Schedule the op
        schedule.ordered_operators.append(op)

        # Update all the ops that depends on "op". Enlist the ones that are
        # ready to be computed
        for output in op.outputs.values():
            if output not in node_to_op:
                continue
            for new_op in node_to_op[output]:
                # "new_op" depends on the result of "op".
                assert new_op in op_to_num_pending_inputs
                num_missing_inputs = op_to_num_pending_inputs[new_op] - 1
                op_to_num_pending_inputs[new_op] = num_missing_inputs
                assert num_missing_inputs >= 0

                if num_missing_inputs == 0:
                    # "new_op" can be computed
                    ready_ops.add(new_op)
                    del op_to_num_pending_inputs[new_op]

    assert not op_to_num_pending_inputs
    return schedule


def has_leak(
    output: EvaluationQuery,
    input: Optional[EvaluationQuery] = None,
) -> bool:
    """Tests if a node depends on a leak operator.

    Tests if a [`Node`][temporian.Node] or collection of nodes depends on the
    only operator that can introduce a future leakage:
    [`tp.leak()`][temporian.leak].

    Single input output example:
        ```python
        >>> a = tp.input_node([("f", float)])
        >>> b = tp.moving_sum(a, 5)
        >>> c = tp.leak(b, 6)
        >>> d = tp.prefix("my_prefix_", c)
        >>> e = tp.moving_sum(d, 7)
        >>> # The computation of "e" contains a leak.
        >>> assert tp.has_leak(e)
        >>> # The computation of "e" given "d" does not contain a leak.
        >>> assert not tp.has_leak(e, d)

        ```

    Args:
        output: Nodes to compute. Supports Node, dict of Nodes and list of
            Nodes.
        input: Optional input nodes. Supports Node, dict of Nodes and list of
            Nodes. If not specified, assumes for the input nodes to be the the
            raw data inputs e.g. [`tp.input_node()`][temporian.input_node] and
            [`tp.event_set()`][temporian.event_set].

    Returns:
        True if and only if the computation of `output` from `inputs` depends
        on a [`tp.leak()`][temporian.leak] operator.
    """

    if input is None:
        normalized_input = None
    else:
        normalized_input = _normalize_query(input)

    normalized_output = _normalize_query(output)

    graph = infer_graph(inputs=normalized_input, outputs=normalized_output)

    for operator in graph.operators:
        if operator.operator_key() == "LEAK":
            return True

    return False


def _normalize_input(input: EvaluationInput) -> Dict[Node, EventSet]:
    """Normalizes an input into a dictionary of node to evsets."""

    if isinstance(input, dict):
        keys_are_node = all([isinstance(x, Node) for x in input.keys()])
        values_are_node = all([isinstance(x, EventSet) for x in input.values()])

        if not keys_are_node or not values_are_node:
            raise ValueError(
                "Invalid input argument. Dictionary input argument should be a"
                f" dictionary of Node to EventSet. Instead, got {input!r}"
            )

        return input

    if isinstance(input, EventSet):
        return {input.node(): input}

    if isinstance(input, list):
        return {evset.node(): evset for evset in input}

    raise TypeError(
        "Evaluate input argument must be an EventSet, list of EventSet, or a"
        f" dictionary of Node to EventSets. Received {input!r} instead."
    )


def _normalize_query(query: EvaluationQuery) -> Set[Node]:
    """Normalizes a query into a list of query Nodes."""

    if isinstance(query, Node):
        return {query}

    if isinstance(query, set):
        return query

    if isinstance(query, list):
        return set(query)

    if isinstance(query, dict):
        return set(query.values())

    raise TypeError(
        f"Evaluate query argument must be one of {EvaluationQuery}."
        f" Received {type(query)} instead."
    )


def _denormalize_outputs(
    outputs: Dict[Node, EventSet], query: EvaluationQuery
) -> EvaluationResult:
    """Converts outputs into the same format as the query."""

    if isinstance(query, Node):
        return outputs[query]

    if isinstance(query, list):
        return [outputs[k] for k in query]

    if isinstance(query, dict):
        return {
            query_key: outputs[query_evt]
            for query_key, query_evt in query.items()
        }

    raise RuntimeError("Unexpected case")
