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

"""Graph class definition and inference logic."""

from __future__ import annotations
from copy import deepcopy
from typing import List, Set, Dict, Union, Optional

from temporian.core.data.node import EventSetNode, Feature, Sampling
from temporian.core.operators import base

NamedEventSetNodes = Union[
    Dict[str, EventSetNode], List[EventSetNode], Set[EventSetNode], EventSetNode
]


class Graph:
    """A set of operators, nodes, features and samplings."""

    def __init__(self):
        self._operators: Set[base.Operator] = set()
        self._features: Set[Feature] = set()
        self._nodes: Set[EventSetNode] = set()
        self._samplings: Set[Sampling] = set()
        self._inputs: Set[EventSetNode] = set()
        self._outputs: Set[EventSetNode] = set()
        self._named_inputs: Optional[Dict[str, EventSetNode]] = None
        self._named_outputs: Optional[Dict[str, EventSetNode]] = None

    @property
    def samplings(self) -> Set[Sampling]:
        return self._samplings

    @property
    def features(self) -> Set[Feature]:
        return self._features

    @property
    def operators(self) -> Set[base.Operator]:
        return self._operators

    @property
    def nodes(self) -> Set[EventSetNode]:
        return self._nodes

    @property
    def inputs(self) -> Set[EventSetNode]:
        return self._inputs

    @property
    def outputs(self) -> Set[EventSetNode]:
        return self._outputs

    @property
    def named_inputs(self) -> Optional[Dict[str, EventSetNode]]:
        return self._named_inputs

    @property
    def named_outputs(self) -> Optional[Dict[str, EventSetNode]]:
        return self._named_outputs

    @named_inputs.setter
    def named_inputs(self, named_inputs: Optional[Dict[str, EventSetNode]]):
        self._named_inputs = named_inputs

    @named_outputs.setter
    def named_outputs(self, named_outputs: Optional[Dict[str, EventSetNode]]):
        self._named_outputs = named_outputs

    def add_operator(self, operator: base.Operator) -> None:
        self._operators.add(operator)

    def add_sampling(self, sampling: Sampling) -> None:
        self._samplings.add(sampling)

    def add_feature(self, feature: Feature) -> None:
        self._features.add(feature)

    def add_node(self, node: EventSetNode) -> None:
        self._nodes.add(node)

    def input_features(self) -> Set[Feature]:
        return {
            feature for node in self.inputs for feature in node.feature_nodes
        }

    def set_input_node_names(self, names: Dict[str, EventSetNode]):
        if self._inputs is not None:
            named_nodes = set(names.values())
            if named_nodes != self._inputs:
                raise ValueError(
                    "All the input nodes and only the input nodes should be"
                    f" renamed. Input nodes: {self._inputs}. Renamed nodes:"
                    f" {names}"
                )

        self._named_inputs = names

    def set_output_node_names(self, names: Dict[str, EventSetNode]):
        if self._outputs is not None:
            named_nodes = set(names.values())
            if named_nodes != self._outputs:
                raise ValueError(
                    "All the input nodes and only the input nodes should be"
                    f" renamed. Input nodes: {self._outputs}. Renamed nodes:"
                    f" {names}"
                )

        self._named_outputs = names

    def input_samplings(self) -> Set[Sampling]:
        return {node.sampling_node for node in self.inputs}

    def __repr__(self):
        s = "Graph\n============\n"

        def p(title, elements):
            nonlocal s
            s += f"{title} ({len(elements)}):\n"
            for e in elements:
                s += f"\t{e}\n"
            s += "\n"

        p("Operators", self.operators)
        p("Features", self.features)
        p("Samplings", self.samplings)
        p("EventSetNodes", self.nodes)

        def p2(title, dictionary):
            nonlocal s
            s += f"{title} ({len(dictionary)}):\n"
            for k, v in dictionary.items():
                s += f"\t{k}:{v}\n"
            s += "\n"

        def p3(title, l):
            nonlocal s
            s += f"{title} ({len(l)}):\n"
            for v in l:
                s += f"\t{v}\n"
            s += "\n"

        if self.named_inputs is not None:
            p2("Named inputs", self.named_inputs)
        else:
            p3("Inputs", self.inputs)

        if self.named_outputs is not None:
            p2("Named output", self.named_outputs)
        else:
            p3("Output", self.outputs)
        return s

    def replace_named_inputs(
        self, named_inputs: Dict[str, EventSetNode]
    ) -> Dict[str, EventSetNode]:
        """Applies the operators in this graph to new inputs.

        Note that the objects in the modified graph are very inconsistent, but
        that's okay since we won't use it anymore. When running it or save it
        the graph will be re-inferred.

        Args:
            named_inputs: The new inputs to the graph.

        Returns:
            The graph's named outputs.
        """

        # Avoid messing with the objects in the received graph
        # Else each time we apply the graph on inputs it changes
        # Might be possible to optimize by copying only what's necessary
        g = deepcopy(self)

        assert g.named_inputs is not None
        assert g.named_outputs is not None

        for name, new_node in named_inputs.items():
            if name not in g.named_inputs:
                raise ValueError(
                    f"Input node {name} is not in the graph's inputs. Inputs:"
                    f" {g.named_inputs}"
                )
            old_node = g.named_inputs[name]
            # Replace node as input in all operators that depend on it
            # TODO: create and maintain a mapping from named_input to operator
            # to make this more efficient
            for operator in g.operators:
                for name, inp in operator.inputs.items():
                    if inp is old_node:
                        operator.inputs[name] = new_node
        return g.named_outputs


def infer_graph_named_nodes(
    inputs: Optional[NamedEventSetNodes], outputs: NamedEventSetNodes
) -> Graph:
    """Extracts the nodes in between the output and input nodes.

    Unlike infer_graph, infer_graph_named_nodes requires for the input and
    output nodes to be named.
    """

    normalized_inputs: Optional[Dict[str, EventSetNode]] = None
    input_nodes = None
    if inputs is not None:
        normalized_inputs = normalize_named_nodes(inputs)
        input_nodes = set(normalized_inputs.values())

    normalized_outputs = normalize_named_nodes(outputs)
    output_nodes = set(normalized_outputs.values())

    g = infer_graph(inputs=input_nodes, outputs=output_nodes)
    if normalized_inputs is None:
        normalized_inputs = normalize_named_nodes(list(g.inputs))

    g.set_input_node_names(normalized_inputs)
    g.set_output_node_names(normalized_outputs)
    return g


def infer_graph(
    inputs: Optional[Set[EventSetNode]], outputs: Set[EventSetNode]
) -> Graph:
    """Extracts the nodes in between the output and input nodes.

    If inputs is set, fails if outputs cannot be computed from `inputs`.
    If inputs is not set, infers the required set of inputs.

    Args:
        inputs: Set of available input nodes. If None, inputs are inferred.
        outputs: Set of expected output nodes.

    Returns:
        The inferred graph.

    Raises:
        ValueError: If there are repeated nodes in the `inputs`; an
            unexpected type of input is provided; an unnamed node is inferred
            as input; or some nodes are required but not provided.
    """
    # The following algorithm lists all the nodes between the output and
    # input nodes. Informally, the algorithm works as follow:
    #
    # pending_node <= use outputs
    # done_node <= empty
    #
    # While pending node not empty:
    #   Extract a node from pending_node
    #   if node is a provided input node
    #       continue
    #   if node has no creator
    #       record this node for future error / input inference
    #       continue
    #   Adds all the input nodes of node's creator op to the pending list

    # # Extract the names
    # outputs_set = outputs if isinstance(outputs, set) else
    #         set(outputs.values())
    # if inputs is None:
    #     input_set = None
    # else:
    #     input_set = inputs if isinstance(inputs, set) else
    #         set(inputs.values())

    graph = Graph()
    graph.outputs.update(outputs)

    # The next nodes to process. EventSetNodes are processed from the outputs to
    # the inputs.
    pending_nodes: Set[EventSetNode] = outputs.copy()

    # Features already processed.
    done_nodes: Set[EventSetNode] = set()

    # List of the missing nodes. Used to create an error message.
    missing_nodes: Set[EventSetNode] = set()

    while pending_nodes:
        # Select a node to process.
        node = next(iter(pending_nodes))
        pending_nodes.remove(node)
        assert node not in done_nodes

        graph.add_node(node)

        if inputs is not None and node in inputs:
            # The feature is provided by the user.
            graph.inputs.add(node)
            continue

        if node.creator is None:
            # The node does not have a source.
            if inputs is not None:
                missing_nodes.add(node)
            else:
                graph.inputs.add(node)
            continue

        # Record the operator op.
        graph.add_operator(node.creator)

        # Add the parent nodes to the pending list.
        for input_node in node.creator.inputs.values():
            if input_node in done_nodes:
                # Already processed.
                continue

            pending_nodes.add(input_node)

        # Record the operator outputs. While the user did not request
        # them, they will be created (and so, we need to track them).
        for output_node in node.creator.outputs.values():
            graph.add_node(output_node)

    if missing_nodes:
        # Fail if not all nodes are sourced.
        raise ValueError(
            "The following input nodes are required but not provided as"
            f" input:\n{missing_nodes}"
        )

    # Record all the features and samplings.
    for e in graph.nodes:
        graph.add_sampling(e.sampling_node)
        for f in e.feature_nodes:
            graph.add_feature(f)

    return graph


def normalize_named_nodes(src: NamedEventSetNodes) -> Dict[str, EventSetNode]:
    """Normalizes a node or list of nodes into a dictionary of nodes."""

    save_src = src

    if isinstance(src, EventSetNode):
        # Will be further processed after.
        src = [src]

    if isinstance(src, set):
        src = list(src)

    if isinstance(src, list):
        new_src = {}
        for node in src:
            if node.name is None:
                raise ValueError(
                    "Input / output node or list nodes need to be named "
                    'with "node.name = ...". Alternatively, provide a '
                    "dictionary of nodes."
                )
            new_src[node.name] = node
        src = new_src

    if not isinstance(src, dict):
        raise ValueError(
            f'Unexpected node(s) "{save_src}". Expecting dict of nodes, '
            "list of nodes, or a single node."
        )
    return src
