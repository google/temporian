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
from typing import Any, Dict, List, Set, Tuple, Union
from collections import defaultdict

from temporian.core.data.node import Node
from temporian.core.operators import base
from temporian.implementation.numpy import evaluation as np_eval
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core import graph


EvaluationQuery = Union[Node, List[Node], Dict[str, Node]]
EvaluationInput = Union[
    # dict of node/node name to corresponding event set
    Dict[graph.NodeInputArg, EventSet],
    # list of event sets, and nodes are associated by name
    List[EventSet],
    # single event set, and node is associated by name
    EventSet,
]
EvaluationResult = Union[EventSet, List[EventSet], Dict[str, EventSet]]


def evaluate(
    query: EvaluationQuery,
    input: EvaluationInput,
    verbose: int = 0,
    check_execution: bool = True,
) -> EvaluationResult:
    """Evaluates nodes on event sets.

    Performs all computation defined by the graph between the `query` nodes and
    the `input` event sets.

    The result is returned in the same format as the `query` argument.

    Args:
        query: Nodes to compute. Supports Node, dict of Nodes and list of Nodes.
        input: Event sets to be used for the computation. Supports EventSet,
            list of EventSets, dict of Nodes to EventSets, and dict of node
            names to event sets. If a single event set or list of event sets,
            they must be named and will be used as input for the nodes with the
            same name. If a dict of node names to event sets, they will be used
            as input for the nodes with those names. If a dict of nodes to event
            sets, they will be used as input for those nodes.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.
        check_execution: If true, the input and output of the op implementation
            are validated to check any bug in the library internal code. If
            false, checks are skipped.

    Returns:
        An object with the same structure as `query` containing the results.
        If `query` is a dictionary of nodes, the returned object will be a
        dictionary of event sets. If `query` is a list of nodes, the
        returned value will be a list of event sets with the same order.
    """
    # TODO: Create an internal configuration object for options such as
    # `check_execution`.

    begin_time = time.perf_counter()

    normalized_query = _normalize_query(query)

    input = _normalize_input(input)

    if verbose >= 1:
        print("Build schedule", file=sys.stderr)

    # Schedule execution
    input_nodes = list(input.keys())
    schedule, names_to_nodes = _build_schedule(
        inputs=input_nodes, outputs=normalized_query, verbose=verbose
    )

    # Replace node names for actual nodes in input
    for name, node in names_to_nodes.items():
        input[node] = input[name]
        del input[name]

    if verbose == 1:
        print(
            f"Run {len(schedule)} operators",
            file=sys.stderr,
        )

    elif verbose >= 2:
        print("Schedule:\n", schedule, file=sys.stderr)

    # Evaluate schedule
    #
    # Note: "outputs" is a dictionary of event (including the query events) to
    # event data.
    outputs = np_eval.evaluate_schedule(
        input,
        schedule,
        verbose=verbose,
        check_execution=check_execution,
    )

    end_time = time.perf_counter()

    if verbose == 1:
        print(f"Execution in {end_time - begin_time:.5f} s", file=sys.stderr)

    return _denormalize_outputs(outputs, query)


def _build_schedule(
    inputs: List[graph.NodeInputArg],
    outputs: List[Node],
    verbose: int = 0,
) -> Tuple[List[base.Operator], Dict[str, Node]]:
    """Calculates which operators need to be executed in which order to compute
    a set of output nodes given a set of input nodes.

    This implementation is based on Kahn's algorithm.

    Args:
        inputs: Input nodes or names of input nodes.
        outputs: Output nodes.
        verbose: If >0, prints details about the execution on the standard error
            output. The larger the number, the more information is displayed.

    Returns:
        Tuple of:
            - Ordered list of operators, such that the first operator should be
            computed before the second, second before the third, etc.
            - Mapping of node name inputs to nodes. The keys are the string
            values in the `inputs` argument, and the values are the nodes
            corresponding to each one. If a value was already a node, it won't
            be present in the returned dictionary.
    """
    # List all nodes and operators in between inputs and outputs.
    #
    # Fails if the outputs cannot be computed from the inputs e.g. some inputs
    # are missing.
    g, names_to_nodes = graph.infer_graph(
        _list_to_dict(inputs), _list_to_dict(outputs)
    )

    if verbose >= 2:
        print("Graph:\n", graph, file=sys.stderr)

    # Sequence of operators to execute. This is the result of the
    # "_build_schedule" function.
    planned_ops: List[base.Operator] = []

    # Operators ready to be computed (i.e. ready to be added to "planned_ops")
    # as all their inputs are already computed by "planned_ops" or specified by
    # "inputs".
    ready_ops: Set[base.Operator] = set()

    # "node_to_op[e]" is the list of operators with node "e" as input.
    node_to_op: Dict[Node, List[base.Operator]] = defaultdict(lambda: [])

    # "op_to_num_pending_inputs[op]" is the number of "not yet scheduled" inputs
    # of operator "op". Operators in "op_to_num_pending_inputs" have not yet
    # scheduled.
    op_to_num_pending_inputs: Dict[base.Operator, int] = defaultdict(lambda: 0)

    # Compute "node_to_op" and "op_to_num_pending_inputs".
    inputs_set = set(inputs)
    for op in g.operators:
        num_pending_inputs = 0
        for input_node in op.inputs.values():
            if input_node in inputs_set or (
                input_node.name and input_node.name in inputs_set
            ):
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
        planned_ops.append(op)

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

    return planned_ops, names_to_nodes


def _normalize_input(
    input: EvaluationInput,
) -> Dict[graph.NodeInputArg, EventSet]:
    """Normalizes an input into a dictionary of node or node names to evsets."""

    if isinstance(input, dict):
        return input

    if isinstance(input, EventSet):
        if not input.name:
            raise ValueError(
                f"{input} must have a name to be used as an unnamed input."
                " Either set its name or pass inputs as a dict."
            )
        return {input.name: input}

    if isinstance(input, list):
        if not all((evset.name for evset in input)):
            raise ValueError(
                f"All event sets in {input} must have a name to be used as"
                " unnamed inputs. Either set their names or pass inputs as a"
                " dict."
            )
        result = {evset.name: evset for evset in input}
        if len(result) < len(input):
            raise ValueError(
                f"Duplicate names in {input}. Input node names must be unique."
            )
        return result

    raise TypeError(
        f"Evaluate input argument must be one of {EvaluationInput}."
        f" Received {type(input)} instead."
    )


def _normalize_query(query: EvaluationQuery) -> List[Node]:
    """Normalizes a query into a list of query nodes."""
    normalized_query: List[Node] = {}

    if isinstance(query, Node):
        # The query is a single value
        normalized_query = [query]

    elif isinstance(query, list):
        # The query is a list
        normalized_query = query

    elif isinstance(query, dict):
        # The query is a dictionary
        normalized_query = list(query.values())

    else:
        # TODO: improve error message
        raise TypeError(
            f"Evaluate query argument must be one of {EvaluationQuery}."
            f" Received {type(query)} instead."
        )

    return normalized_query


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


def _list_to_dict(l: List[Any]) -> Dict[str, Any]:
    """Converts a list into a dict with a text index key."""
    return {str(i): x for i, x in enumerate(l)}
