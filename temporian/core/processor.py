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

"""Processor class definition and inference logic."""

from typing import List, Set, Dict, Tuple, Union, Optional

from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base

MultipleNodeArg = Union[Dict[str, Node], List[Node], Node]
NodeInputArg = Union[Node, str]


class Processor(object):
    """A set of operators, nodes, features and samplings."""

    def __init__(self):
        self._operators: Set[base.Operator] = set()
        self._features: Set[Feature] = set()
        self._nodes: Set[Node] = set()
        self._samplings: Set[Sampling] = set()
        self._inputs: Dict[str, Node] = {}
        self._outputs: Dict[str, Node] = {}

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
    def nodes(self) -> Set[Node]:
        return self._nodes

    @property
    def inputs(self) -> Dict[str, Node]:
        return self._inputs

    @property
    def outputs(self) -> Dict[str, Node]:
        return self._outputs

    def add_operator(self, operator: base.Operator) -> None:
        self._operators.add(operator)

    def add_sampling(self, sampling: Sampling) -> None:
        self._samplings.add(sampling)

    def add_feature(self, feature: Feature) -> None:
        self._features.add(feature)

    def add_node(self, node: Node) -> None:
        self._nodes.add(node)

    @inputs.setter
    def inputs(self, inputs: Dict[str, Node]) -> None:
        self._inputs = inputs

    @outputs.setter
    def outputs(self, outputs: Dict[str, Node]) -> None:
        self._outputs = outputs

    def input_features(self) -> Set[Feature]:
        return {
            feature
            for node in self.inputs.values()
            for feature in node.features
        }

    def input_samplings(self) -> Set[Sampling]:
        return {node.sampling for node in self.inputs.values()}

    def __repr__(self):
        s = "Processor\n============\n"

        def p(title, elements):
            nonlocal s
            s += f"{title} ({len(elements)}):\n"
            for e in elements:
                s += f"\t{e}\n"
            s += "\n"

        p("Operators", self.operators)
        p("Features", self.features)
        p("Samplings", self.samplings)
        p("Nodes", self.nodes)

        def p2(title, dictionary):
            nonlocal s
            s += f"{title} ({len(dictionary)}):\n"
            for k, v in dictionary.items():
                s += f"\t{k}:{v}\n"
            s += "\n"

        p2("Inputs", self.inputs)
        p2("Output", self.outputs)
        return s


def infer_processor(
    inputs: Optional[Dict[str, NodeInputArg]],
    outputs: Dict[str, Node],
) -> Tuple[Processor, Dict[str, Node]]:
    """Extracts all the objects between the output and input nodes.

    Fails if any inputs are missing.

    Args:
        inputs: Mapping of strings to input nodes or node names. If None, the
            inputs are inferred. In this case, input nodes have to be named.
        outputs: Output nodes.

    Returns:
        Tuple of:
            - Inferred processor.
            - Mapping of node name inputs to nodes. The keys are the string
            values in the `inputs` argument, and the values are the nodes
            corresponding to each one. If a value was already a node, it won't
            be present in the returned dictionary.

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

    p = Processor()
    p.outputs = outputs

    # The next node to process. Nodes are processed from the outputs to
    # the inputs.
    pending_nodes: Set[Node] = set()
    pending_nodes.update(outputs.values())

    # Index the input nodes for fast retrieval.
    input_nodes: Set[Node] = set()

    # Index the input node names for fast retrieval.
    # Use dict instead of set since we will need their key.
    input_node_names_to_idx_str: Dict[str, str] = {}

    # Create map of node/node name inputs to corresponding node.
    names_to_nodes: Dict[str, Node] = {}

    if inputs is not None:
        p.inputs: Dict[str, Node] = {}

        for idx_str, node_or_name in inputs.items():
            if isinstance(node_or_name, Node):
                # Fail if same node is passed as input twice.
                if node_or_name in input_nodes:
                    raise ValueError(
                        f"Duplicate nodes in {inputs}. Input nodes must be"
                        " unique."
                    )
                input_nodes.add(node_or_name)

                # Only add node_or_name to p.inputs if its a node
                # If a name, it will be added to p.inputs and names_to_nodes
                # when we find it while traversing the graph.
                p.inputs[idx_str] = node_or_name

            elif isinstance(node_or_name, str):
                # Fail if same node is passed as input twice.
                # Need to check name against other input names and against names
                # of input nodes.
                if (
                    node_or_name
                    in {node.name for node in input_nodes if node.name}
                    or node_or_name in input_node_names_to_idx_str
                ):
                    raise ValueError(
                        f"Duplicate nodes or names in {inputs}. Input nodes and"
                        " names must be unique."
                    )
                input_node_names_to_idx_str[node_or_name] = idx_str

            else:
                raise ValueError(
                    f"Unexpected input {node_or_name}. "
                    "Expecting a node or node name."
                )

    # Features already processed.
    done_nodes: Set[Node] = set()

    # List of the missing nodes. They will be used to infer the input features
    # (if infer_inputs=True), or to raise an error (if infer_inputs=False).
    missing_nodes: Set[Node] = set()

    while pending_nodes:
        # Select a node to process.
        node = next(iter(pending_nodes))
        pending_nodes.remove(node)
        assert node not in done_nodes

        p.add_node(node)

        if node in input_nodes:
            # The feature is provided by the user.
            continue

        elif node.name and node.name in input_node_names_to_idx_str:
            # The feature is provided by the user.
            # Add node to p.inputs and names_to_nodes now that we have it.
            idx_str = input_node_names_to_idx_str[node.name]
            p.inputs[idx_str] = node
            names_to_nodes[node.name] = node
            continue

        if node.creator is None:
            # The node does not have a source.
            missing_nodes.add(node)
            continue

        # Record the operator.
        p.add_operator(node.creator)

        # Add the parent nodes to the pending list.
        for input_node in node.creator.inputs.values():
            if input_node in done_nodes:
                # Already processed.
                continue

            pending_nodes.add(input_node)

        # Record the operator outputs. While the user did not request
        # them, they will be created (and so, we need to track them).
        for output_node in node.creator.outputs.values():
            p.add_node(output_node)

    if inputs is None:
        # Infer the inputs
        infered_inputs: Dict[str, Node] = {}
        for node in missing_nodes:
            if node.name is None:
                raise ValueError(f"Cannot infer input on unnamed node {node}")
            infered_inputs[node.name] = node
        p.inputs = infered_inputs

    else:
        # Fail if not all nodes are sourced.
        if missing_nodes:
            raise ValueError(
                "Some nodes are required but "
                f"not provided as input:\n {missing_nodes}"
            )

    # Record all the features and samplings.
    for e in p.nodes:
        p.add_sampling(e.sampling)
        for f in e.features:
            p.add_feature(f)

    return p, names_to_nodes


def normalize_multiple_node_arg(src: MultipleNodeArg) -> Dict[str, Node]:
    """Normalizes a node or list of nodes into a dictionary of nodes."""

    save_src = src

    if isinstance(src, Node):
        src = [src]

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
