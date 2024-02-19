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
from typing import Dict, List, Set, Optional
from collections import defaultdict

from temporian.core.data.node import EventSetNode
from temporian.core.operators.base import Operator
from temporian.core.typing import (
    EventSetCollection,
    EventSetNodeCollection,
    NodeToEventSetMapping,
)
from temporian.implementation.numpy import evaluation as np_eval
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core.graph import infer_graph
from temporian.core.schedule import Schedule, ScheduleStep
from temporian.core.operators.leak import LeakOperator


def run(
    query: EventSetNodeCollection,
    input: NodeToEventSetMapping,
    verbose: int = 0,
    check_execution: bool = True,
) -> EventSetCollection:
    """Evaluates [`EventSetNodes`][temporian.EventSetNode] on [`EventSets`][temporian.EventSet].

    Performs all computation defined by the graph between the `query` EventSetNodes and
    the `input` EventSets.

    The result is returned in the same format as the `query` argument.

    Single input output example:
        ```python
        >>> input_evset = tp.event_set(timestamps=[1, 2, 3], features={"f": [0, 4, 10]})
        >>> input_node = input_evset.node()
        >>> output_node = input_node.moving_sum(5)
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
        >>> step_2 = step_1.simple_moving_average(2)

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
        query: EventSetNodes to compute. Supports EventSetNode, dict of EventSetNodes and list of EventSetNodes.
        input: Event sets to be used for the computation. Supports EventSet,
            list of EventSets, dict of EventSetNodes to EventSets, and dict of EventSetNode
            names to EventSet. If a single EventSet or list of EventSet,
            they must be named and will be used as input for the EventSetNodes with the
            same name. If a dict of EventSetNode names to EventSet, they will be used
            as input for the EventSetNodes with those names. If a dict of EventSetNodes to event
            sets, they will be used as input for those EventSetNodes.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.
        check_execution: If true, the input and output of the op implementation
            are validated to check any bug in the library internal code. If
            false, checks are skipped.

    Returns:
        An object with the same structure as `query` containing the results.
            If `query` is a dictionary of EventSetNodes, the returned object will be a
            dictionary of EventSet. If `query` is a list of EventSetNodes, the
            returned value will be a list of EventSet with the same order.
    """
    # TODO: Create an internal configuration object for options such as
    # `check_execution`.

    begin_time = time.perf_counter()

    input = _normalize_input(input)
    normalized_query = _normalize_query(query)

    if verbose >= 1:
        print("Build schedule", file=sys.stderr, flush=True)

    # Schedule execution
    assert isinstance(normalized_query, set)
    input_nodes = set(input.keys())
    schedule = build_schedule(
        inputs=input_nodes, outputs=normalized_query, verbose=verbose
    )

    if verbose == 1:
        print(
            f"Run {len(schedule.steps)} operators",
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
    inputs: Optional[Set[EventSetNode]],
    outputs: Set[EventSetNode],
    verbose: int = 0,
) -> Schedule:
    """Calculates which operators need to be executed in which order to compute
    a set of output EventSetNodes given a set of input EventSetNodes.

    This implementation is based on Kahn's algorithm.

    Args:
        inputs: Input EventSetNodes.
        outputs: Output EventSetNodes.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.

    Returns:
        Tuple of:
            - Ordered list of operators, such that the first operator should be
            computed before the second, second before the third, etc.
            - Mapping of EventSetNode name inputs to EventSetNodes. The keys are the string
            values in the `inputs` argument, and the values are the EventSetNodes
            corresponding to each one. If a value was already an EventSetNode, it won't
            be present in the returned dictionary.
    """
    # List all EventSetNodes and operators in between inputs and outputs.
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
    ready_ops: List[Operator] = []
    ready_ops_set: Set[Operator] = set()

    # "node_to_op[e]" is the list of operators with node "e" as input.
    node_to_op: Dict[EventSetNode, List[Operator]] = defaultdict(lambda: [])

    # "op_to_num_pending_inputs[op]" is the number of "not yet scheduled" inputs
    # of operator "op". Operators in "op_to_num_pending_inputs" have not yet
    # scheduled.
    op_to_num_pending_inputs: Dict[Operator, int] = defaultdict(lambda: 0)

    # Compute "node_to_op" and "op_to_num_pending_inputs".
    for op in graph.operators:
        num_pending_inputs = 0
        for input_node in op.inputs.values():
            node_to_op[input_node].append(op)
            if input_node in graph.inputs:
                # This input is already available
                continue
            num_pending_inputs += 1
        if num_pending_inputs == 0:
            # Ready to be scheduled
            ready_ops.append(op)
            ready_ops_set.add(op)
        else:
            # Some of the inputs are missing.
            op_to_num_pending_inputs[op] = num_pending_inputs

    # Make evaluation order deterministic.
    #
    # Execute the op with smallest internal ordered id first.
    ready_ops.sort(key=lambda op: op._internal_ordered_id, reverse=True)

    # Compute the schedule
    while ready_ops:
        # Get an op ready to be scheduled
        op = ready_ops.pop()
        ready_ops_set.remove(op)

        # Nodes released after the op is executed
        released_nodes = []
        for input in op.inputs.values():
            if input in outputs:
                continue
            if input not in node_to_op:
                continue
            # The list of ops that depends on this input (including the current
            # op "op").
            input_usage = node_to_op[input]
            input_usage.remove(op)

            if not input_usage:
                released_nodes.append(input)
                del node_to_op[input]

        # Schedule the op
        schedule.steps.append(
            ScheduleStep(op=op, released_nodes=released_nodes)
        )

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
                    ready_ops.append(new_op)
                    ready_ops_set.add(new_op)
                    del op_to_num_pending_inputs[new_op]

    assert not op_to_num_pending_inputs
    return schedule


def has_leak(
    output: EventSetNodeCollection,
    input: Optional[EventSetNodeCollection] = None,
) -> bool:
    """Tests if a node depends on a leak operator.

    Tests if a [`EventSetNode`][temporian.EventSetNode] or collection of nodes
    depends on the only operator that can introduce future leakage:
    [`EventSet.leak()`][temporian.EventSet.leak].

    Single input output example:
        ```python
        >>> a = tp.input_node([("f", tp.float32)])
        >>> b = a.moving_sum(5)
        >>> c = b.leak(6)
        >>> d = c.prefix("my_prefix_")
        >>> e = d.moving_sum(7)
        >>> # The computation of "e" contains a leak.
        >>> tp.has_leak(e)
        True
        >>> # The computation of "e" given "d" does not contain a leak.
        >>> tp.has_leak(e, d)
        False

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
        on a [`EventSet.leak()`][temporian.EventSet.leak] operator.
    """

    if input is None:
        normalized_input = None
    else:
        normalized_input = _normalize_query(input)

    normalized_output = _normalize_query(output)

    graph = infer_graph(inputs=normalized_input, outputs=normalized_output)

    leak_key = LeakOperator.operator_key()
    for operator in graph.operators:
        if operator.operator_key() == leak_key:
            return True

    return False


def _normalize_input(
    input: NodeToEventSetMapping,
) -> Dict[EventSetNode, EventSet]:
    """Normalizes an input into a dictionary of node to evsets."""

    if isinstance(input, dict):
        keys_are_node = all([isinstance(x, EventSetNode) for x in input.keys()])
        values_are_node = all([isinstance(x, EventSet) for x in input.values()])

        if not keys_are_node or not values_are_node:
            raise ValueError(
                "Invalid input argument. Dictionary input argument should be a"
                " dictionary of EventSetNode to EventSet. Instead, got"
                f" {input!r}"
            )

        return input

    if isinstance(input, EventSet):
        return {input.node(): input}

    if isinstance(input, list):
        return {evset.node(): evset for evset in input}

    raise TypeError(
        "Evaluate input argument must be an EventSet, list of EventSet, or a"
        f" dictionary of EventSetNode to EventSets. Received {input!r} instead."
    )


def _normalize_query(query: EventSetNodeCollection) -> Set[EventSetNode]:
    """Normalizes a query into a list of query EventSetNodes."""

    if isinstance(query, EventSetNode):
        return {query}

    if isinstance(query, set):
        return query

    if isinstance(query, list):
        return set(query)

    if isinstance(query, dict):
        return set(query.values())

    raise TypeError(
        f"Evaluate query argument must be one of {EventSetNodeCollection}."
        f" Received {type(query)} instead."
    )


def _denormalize_outputs(
    outputs: Dict[EventSetNode, EventSet], query: EventSetNodeCollection
) -> EventSetCollection:
    """Converts outputs into the same format as the query."""

    if isinstance(query, EventSetNode):
        return outputs[query]

    if isinstance(query, list):
        return [outputs[k] for k in query]

    if isinstance(query, dict):
        return {
            query_key: outputs[query_evt]
            for query_key, query_evt in query.items()
        }

    raise RuntimeError("Unexpected case")
