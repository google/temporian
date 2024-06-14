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

"""Serialization/unserialization of a graph and its components."""

import inspect
import logging
from typing import (
    Callable,
    List,
    Set,
    Any,
    Dict,
    Tuple,
    Optional,
    Mapping,
    Union,
)

from google.protobuf import text_format

from temporian.core import operator_lib
from temporian.core import graph
from temporian.core.data.node import (
    EventSetNode,
    Sampling,
    Feature,
    input_node,
)
from temporian.core.data.schema import Schema
from temporian.core.compilation import compile
from temporian.core.operators import base
from temporian.core.data.dtype import DType
from temporian.core.typing import (
    EventSetOrNode,
    NormalizedIndexKey,
    NormalizedIndexKeyItem,
)
from temporian.implementation.numpy.data.event_set import (
    EventSet,
)
from temporian.proto import core_pb2 as pb

DTYPE_MAPPING = {
    DType.FLOAT64: pb.DType.DTYPE_FLOAT64,
    DType.FLOAT32: pb.DType.DTYPE_FLOAT32,
    DType.INT64: pb.DType.DTYPE_INT64,
    DType.INT32: pb.DType.DTYPE_INT32,
    DType.BOOLEAN: pb.DType.DTYPE_BOOLEAN,
    DType.STRING: pb.DType.DTYPE_STRING,
}
INV_DTYPE_MAPPING = {v: k for k, v in DTYPE_MAPPING.items()}


# TODO: allow saved fn to return a list or single node too
def save(
    fn: Callable[..., Dict[str, EventSetOrNode]],
    path: str,
    *args: Union[EventSetNode, EventSet, Schema],
    **kwargs: Union[EventSetNode, EventSet, Schema],
) -> None:
    """Saves a compiled Temporian function to a file.

    The saved function must only take
    [`EventSetOrNodes`][temporian.types.EventSetOrNode] as arguments, return a
    dictionary of names to EventSetOrNodes, and be decorated with
    [`@tp.compile`][temporian.compile].

    Temporian saves the graph built between the function's input and output
    nodes, not the function itself. Any arbitrary code that
    is executed in the function will not be ran when loading it back up and
    executing it.

    If you need to save a function that additionally takes other types of
    arguments, try using `functools.partial` to create a new function that takes
    only EventSetNodes, and save that instead. Note that the partial function
    needs to be compiled too, with `tp.compile(partial(...))`.

    Example:
        ```python
        >>> a = tp.event_set([0, 1, 2], {"a": [10.0, 20.0, 10.0]})

        >>> # define function and compile it
        >>> @tp.compile()
        ... def func(evset):
        ...     return {"avg_evset": evset.simple_moving_average(3)}

        >>> # save function
        >>> file_path = tmp_dir / "my_prep.tem"
        >>> tp.save(func, file_path, evset=a)

        >>> # load function
        >>> loaded_func = tp.load(file_path)
        >>> loaded_func(a)
        {'avg_evset': indexes: []
        features: [('a', float64)]
        events:
            (3 events):
                timestamps: [0. 1. 2.]
                'a': [10.     15.     13.3333]
        ...

        ```

    Args:
        fn: The function to save.
        path: The path to save the function to.
        args: Positional arguments to pass to the function to trace it. The
            arguments can be either EventSets, EventSetNodes, or raw Schemas. In
            all cases, the values will be converted to EventSetNodes before
            being passed to the function to trace it.
        kwargs: Keyword arguments to pass to the function to trace it. Same
            restrictions as for `args`.

    Raises:
        ValueError: If the received function is not compiled.
        ValueError: If any of the received inputs is not of the specified types.
        ValueError: If the function doesn't return the specified type.
    """
    if not hasattr(fn, "is_tp_compiled"):
        raise ValueError(
            "Can only save a function that has been compiled with"
            " `@tp.compile`."
        )

    merged_kwargs = _kwargs_from_args_and_kwargs(
        list(inspect.signature(fn).parameters.keys()), args, kwargs
    )
    node_kwargs = {k: _process_fn_input(v) for k, v in merged_kwargs.items()}

    outputs = fn(**node_kwargs)

    _check_fn_outputs(outputs)

    save_graph(inputs=node_kwargs, outputs=outputs, path=path)  # type: ignore


def load(
    path: str,
) -> Callable[..., Dict[str, EventSetNode]]:
    """Loads a compiled Temporian function from a file.

    The loaded function receives the same positional and keyword arguments and
    applies the same operator graph to its inputs as when it was saved.

    Example:
        ```python
        >>> a = tp.event_set([0, 1, 2], {"a": [10.0, 20.0, 10.0]})

        >>> # define function and compile it
        >>> @tp.compile()
        ... def func(evset):
        ...     return {"avg_evset": evset.simple_moving_average(3)}

        >>> # save function
        >>> file_path = tmp_dir / "my_prep.tem"
        >>> tp.save(func, file_path, evset=a)

        >>> # load function
        >>> loaded_func = tp.load(file_path)
        >>> loaded_func(a)
        {'avg_evset': indexes: []
        features: [('a', float64)]
        events:
            (3 events):
                timestamps: [0. 1. 2.]
                'a': [10.     15.     13.3333]
        ...

        ```

    Args:
        path: The path to load the function from.

    Returns:
        The loaded function.
    """
    g = _load_graph(path)

    inputs = g.named_inputs
    assert inputs is not None

    input_names = list(inputs.keys())

    @compile
    def fn(
        *args: EventSetNode,
        **kwargs: EventSetNode,
    ) -> Dict[str, EventSetNode]:
        kwargs = _kwargs_from_args_and_kwargs(input_names, args, kwargs)
        return g.apply_on_inputs(named_inputs=kwargs)

    fn.__signature__ = inspect.signature(fn).replace(
        parameters=[
            inspect.Parameter(
                name=k,
                annotation=EventSetNode,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for k in inputs
        ]
    )

    return fn


def save_graph(
    inputs: Optional[graph.NamedEventSetNodes],
    outputs: graph.NamedEventSetNodes,
    path: str,
) -> None:
    """Saves the graph between the `inputs` and `outputs`
    [`EventSetNodes`][temporian.EventSetNode] to a file.

    Usage example:
        ```python
        >>> evset = tp.event_set(
        ...    timestamps=[1, 2, 3],
        ...    features={"input_feature": [0, 42, 10]}
        ... )

        >>> # Create a graph
        >>> a = evset.node()
        >>> b = a.moving_sum(2)
        >>> b = b.rename("result_feature")

        >>> # Check evaluation
        >>> b.run({a: evset})
        indexes: []
        features: [('result_feature', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'result_feature': [ 0 42 52]
        ...


        >>> # Save the graph
        >>> file_path = tmp_dir / "graph.tem"
        >>> tp.save_graph(
        ...     inputs={"input_node": a},
        ...     outputs={"output_node": b},
        ...     path=file_path,
        ... )

        >>> # Load the graph
        >>> inputs, outputs = tp.load_graph(path=file_path)

        >>> # Evaluate reloaded graph
        >>> a_reloaded = inputs["input_node"]
        >>> b_reloaded = outputs["output_node"]
        >>> b_reloaded.run({a_reloaded: evset})
        indexes: []
        features: [('result_feature', int64)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'result_feature': [ 0 42 52]
        ...

        ```

    Args:
        inputs: Input EventSetNodes. If None, the inputs is inferred. In this case,
            input EventSetNodes have to be named.
        outputs: Output EventSetNodes.
        path: File path to save to.
    """

    # TODO: Add support for compressed / binary serialization.

    g = graph.infer_graph_named_nodes(inputs=inputs, outputs=outputs)

    proto = _serialize(g)
    with open(path, "wb") as f:
        f.write(text_format.MessageToBytes(proto))


def load_graph(
    path: str, squeeze: bool = False
) -> Tuple[
    Union[EventSetNode, Dict[str, EventSetNode]],
    Union[EventSetNode, Dict[str, EventSetNode]],
]:
    """Loads a Temporian graph from a file.

    See [`tp.save()`][temporian.save] and
    [`tp.save_graph()`][temporian.save_graph] for usage examples.

    Args:
        path: File path to load from.
        squeeze: If true, and if the input/output contains a single EventSetNode,
            returns an EventSetNode (instead of a dictionary of EventSetNodes).

    Returns:
        Input and output EventSetNodes.
    """
    g = _load_graph(path=path)

    inputs = g.named_inputs
    outputs = g.named_outputs

    assert inputs is not None
    assert outputs is not None

    if squeeze and len(inputs) == 1:
        inputs = list(inputs.values())[0]

    if squeeze and len(outputs) == 1:
        outputs = list(outputs.values())[0]

    return inputs, outputs


def _load_graph(
    path: str,
) -> graph.Graph:
    with open(path, "rb") as f:
        proto = text_format.Parse(f.read(), pb.Graph())
    g = _unserialize(proto)

    return g


def _process_fn_input(input: Any) -> EventSetNode:
    if isinstance(input, Schema):
        # TODO: Add support for receiving schema directly in input_node
        return input_node(
            features=[(f.name, f.dtype) for f in input.features],
            indexes=[(i.name, i.dtype) for i in input.indexes],
            is_unix_timestamp=input.is_unix_timestamp,
        )
    if isinstance(input, EventSet):
        return input.node()
    if isinstance(input, EventSetNode):
        return input
    raise ValueError(
        "The function's parameters can only be either EventSets, EventSetNodes,"
        " or Schemas to save it."
    )


def _check_fn_outputs(output: Any):
    if not isinstance(output, dict) or not all(
        isinstance(v, EventSetNode) for v in output.values()
    ):
        raise ValueError(
            "The function must return a single EventSetNode or a dictionary"
            " mapping output names to EventSetNodes."
        )


def _kwargs_from_args_and_kwargs(
    param_names: List[str],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Merges args and kwargs into a single name->value param dict."""
    if len(args) > len(param_names):
        raise ValueError(
            f"The function takes {len(param_names)} arguments, but"
            f" {len(args)} positional arguments were received."
        )

    # zip stops at the shortest iterable, extra param names are ignored
    arg_kwargs = {k: v for k, v in zip(param_names, args)}
    for k in arg_kwargs:
        if k in kwargs:
            raise ValueError(
                f"The function received multiple values for the argument {k}."
            )

    return {**arg_kwargs, **kwargs}


def _serialize(src: graph.Graph) -> pb.Graph:
    """Serializes a graph into a protobuffer."""

    if src.named_inputs is None:
        raise ValueError(
            "Cannot serialize a graph without named input EventSetNodes"
        )
    if src.named_outputs is None:
        raise ValueError(
            "Cannot serialize a graph without named output EventSetNodes"
        )

    return pb.Graph(
        operators=[_serialize_operator(o) for o in src.operators],
        nodes=[_serialize_node(e, src.operators) for e in src.nodes],
        features=[_serialize_feature(f, src.operators) for f in src.features],
        samplings=[
            _serialize_sampling(s, src.operators) for s in src.samplings
        ],
        inputs=[
            _serialize_io_signature(k, e) for k, e in src.named_inputs.items()
        ],
        outputs=[
            _serialize_io_signature(k, e) for k, e in src.named_outputs.items()
        ],
    )


def _unserialize(src: pb.Graph) -> graph.Graph:
    """Unserializes a protobuffer into a graph."""

    # Decode the components.
    # All the fields except for the "creator" ones are set.
    samplings = {s.id: _unserialize_sampling(s) for s in src.samplings}
    features = {f.id: _unserialize_feature(f) for f in src.features}
    nodes = {e.id: _unserialize_node(e, samplings, features) for e in src.nodes}
    operators = {o.id: _unserialize_operator(o, nodes) for o in src.operators}

    # Set the creator fields.
    def get_creator(op_id: str) -> base.Operator:
        if op_id not in operators:
            logging.info(operators)
            raise ValueError(f"Non existing creator operator {op_id}")
        return operators[op_id]

    for src_node in src.nodes:
        if src_node.creator_operator_id:
            nodes[src_node.id].creator = get_creator(
                src_node.creator_operator_id
            )
    for src_feature in src.features:
        if src_feature.creator_operator_id:
            features[src_feature.id].creator = get_creator(
                src_feature.creator_operator_id
            )
    for src_sampling in src.samplings:
        if src_sampling.creator_operator_id:
            samplings[src_sampling.id].creator = get_creator(
                src_sampling.creator_operator_id
            )

    # Copy extracted items.
    g = graph.Graph()
    for sampling in samplings.values():
        g.samplings.add(sampling)
    for node in nodes.values():
        g.nodes.add(node)
    for feature in features.values():
        g.features.add(feature)
    for operator in operators.values():
        g.operators.add(operator)

    # IO Signature
    def get_node(node_id: str) -> EventSetNode:
        if node_id not in nodes:
            raise ValueError(f"Non existing node {node_id}")
        return nodes[node_id]

    g.named_inputs = {}
    g.named_outputs = {}

    for item in src.inputs:
        node = get_node(item.node_id)
        g.inputs.add(node)
        g.named_inputs[item.key] = node

    for item in src.outputs:
        node = get_node(item.node_id)
        g.outputs.add(get_node(item.node_id))
        g.named_outputs[item.key] = node

    return g


def _identifier(item: Any) -> str:
    """Creates a unique identifier for an object within a graph."""
    if item is None:
        raise ValueError("Cannot get id of None")
    return str(id(item))


def _identifier_or_none(
    item: Any, options: Optional[Set[Any]] = None
) -> Optional[str]:
    """Creates a unique identifier for an object within a graph."""
    if item is None:
        return None
    if options is not None and item not in options:
        return None
    return str(id(item))


def _all_identifiers(collection: Any) -> Set[str]:
    """Builds the set of identifiers of a collections of nodes/features/..."""
    return {_identifier(x) for x in collection}


def _serialize_operator(src: base.Operator) -> pb.Operator:
    if not src.definition.is_serializable:
        raise ValueError(f"{src.definition.key} operator is not serializable.")

    return pb.Operator(
        id=_identifier(src),
        operator_def_key=src.definition.key,
        inputs=[
            pb.Operator.EventSetNodeArgument(key=k, node_id=_identifier(v))
            for k, v in src.inputs.items()
        ],
        outputs=[
            pb.Operator.EventSetNodeArgument(key=k, node_id=_identifier(v))
            for k, v in src.outputs.items()
        ],
        attributes=[
            _attribute_to_proto(k, v) for k, v in src.attributes.items()
        ],
    )


def _unserialize_operator(
    src: pb.Operator, nodes: Dict[str, EventSetNode]
) -> base.Operator:
    operator_class = operator_lib.get_operator_class(src.operator_def_key)

    def get_node(key):
        if key not in nodes:
            raise ValueError(f"Non existing node {key}")
        return nodes[key]

    input_args = {x.key: get_node(x.node_id) for x in src.inputs}
    output_args = {x.key: get_node(x.node_id) for x in src.outputs}
    attribute_args = {x.key: _attribute_from_proto(x) for x in src.attributes}

    # We construct the operator.
    op: base.Operator = operator_class(**input_args, **attribute_args)

    # Check that the operator signature matches the expected one.
    if op.inputs.keys() != input_args.keys():
        raise ValueError(
            f"Restoring the operator {src.operator_def_key} lead "
            "to an unexpected input signature. "
            f"Expected: {input_args.keys()} Effective: {op.inputs.keys()}"
        )

    if op.outputs.keys() != output_args.keys():
        raise ValueError(
            f"Restoring the operator {src.operator_def_key} lead "
            "to an unexpected output signature. "
            f"Expected: {output_args.keys()} Effective: {op.outputs.keys()}"
        )

    if op.attributes.keys() != attribute_args.keys():
        raise ValueError(
            f"Restoring the operator {src.operator_def_key} lead to an"
            " unexpected attributes signature. Expected:"
            f" {attribute_args.keys()} Effective: {op.attributes.keys()}"
        )
    # TODO: Deep check of equality of the inputs / outputs / attributes

    # Override the inputs / outputs / attributes
    op.inputs = input_args
    op.outputs = output_args
    op.attributes = attribute_args
    return op


def _serialize_node(
    src: EventSetNode, operators: Set[base.Operator]
) -> pb.EventSetNode:
    assert len(src.schema.features) == len(src.feature_nodes)
    logging.info(operators)
    return pb.EventSetNode(
        id=_identifier(src),
        sampling_id=_identifier(src.sampling_node),
        feature_ids=[_identifier(f) for f in src.feature_nodes],
        name=src.name,
        creator_operator_id=(_identifier_or_none(src.creator, operators)),
        schema=_serialize_schema(src.schema),
    )


def _unserialize_node(
    src: pb.EventSetNode,
    samplings: Dict[str, Sampling],
    features: Dict[str, Feature],
) -> EventSetNode:
    if src.sampling_id not in samplings:
        raise ValueError(f"Non existing sampling {src.sampling_id} in {src}")

    def get_feature(key):
        if key not in features:
            raise ValueError(f"Non existing feature {key}")
        return features[key]

    node = EventSetNode(
        schema=_unserialize_schema(src.schema),
        features=[get_feature(f) for f in src.feature_ids],
        sampling=samplings[src.sampling_id],
        name=src.name,
        creator=None,
    )

    assert len(node.schema.features) == len(node.feature_nodes)
    return node


def _serialize_feature(
    src: Feature, operators: Set[base.Operator]
) -> pb.EventSetNode.Feature:
    return pb.EventSetNode.Feature(
        id=_identifier(src),
        creator_operator_id=(_identifier_or_none(src.creator, operators)),
    )


def _unserialize_feature(src: pb.EventSetNode.Feature) -> Feature:
    return Feature(creator=None)


def _serialize_sampling(
    src: Sampling, operators: Set[base.Operator]
) -> pb.EventSetNode.Sampling:
    return pb.EventSetNode.Sampling(
        id=_identifier(src),
        creator_operator_id=(_identifier_or_none(src.creator, operators)),
    )


def _unserialize_sampling(src: pb.EventSetNode.Sampling) -> Sampling:
    return Sampling(creator=None)


def _serialize_schema(src: Schema) -> pb.Schema:
    return pb.Schema(
        features=[
            pb.Schema.Feature(
                name=feature.name,
                dtype=_serialize_dtype(feature.dtype),
            )
            for feature in src.features
        ],
        indexes=[
            pb.Schema.Index(
                name=index.name,
                dtype=_serialize_dtype(index.dtype),
            )
            for index in src.indexes
        ],
        is_unix_timestamp=src.is_unix_timestamp,
    )


def _unserialize_schema(src: pb.Schema) -> Schema:
    return Schema(
        features=[
            (feature.name, _unserialize_dtype(feature.dtype))
            for feature in src.features
        ],
        indexes=[
            (index.name, _unserialize_dtype(index.dtype))
            for index in src.indexes
        ],
        is_unix_timestamp=src.is_unix_timestamp,
    )


def _serialize_dtype(dtype) -> pb.DType:
    if dtype not in DTYPE_MAPPING:
        raise ValueError(f"Non supported type {dtype}")
    return DTYPE_MAPPING[dtype]


def _unserialize_dtype(dtype: pb.DType):
    if dtype not in INV_DTYPE_MAPPING:
        raise ValueError(f"Non supported type {dtype}")
    return INV_DTYPE_MAPPING[dtype]


def _serialize_index_key_item(
    item: NormalizedIndexKeyItem,
) -> pb.Operator.Attribute.ListIndexKeys.IndexKey.IndexKeyItem:
    if isinstance(item, bytes):
        return pb.Operator.Attribute.ListIndexKeys.IndexKey.IndexKeyItem(
            bytes_=item
        )
    if isinstance(item, int):
        return pb.Operator.Attribute.ListIndexKeys.IndexKey.IndexKeyItem(
            integer_64=item
        )
    raise ValueError(f"Non supported index key item: {item}")


def _unserialize_index_key_item(
    item: pb.Operator.Attribute.ListIndexKeys.IndexKey.IndexKeyItem,
) -> NormalizedIndexKeyItem:
    if item.HasField("bytes_"):
        return item.bytes_
    if item.HasField("integer_64"):
        return item.integer_64
    raise ValueError(f"Non supported index key item: {item}")


def _serialize_index_key(
    key: NormalizedIndexKey,
) -> pb.Operator.Attribute.ListIndexKeys.IndexKey:
    return pb.Operator.Attribute.ListIndexKeys.IndexKey(
        values=[_serialize_index_key_item(x) for x in key]
    )


def _unserialize_index_key(
    key: pb.Operator.Attribute.ListIndexKeys.IndexKey,
) -> NormalizedIndexKey:
    return tuple(_unserialize_index_key_item(x) for x in key.values)


def _attribute_to_proto(
    key: str, value: base.AttributeType
) -> pb.Operator.Attribute:
    if isinstance(value, str):
        return pb.Operator.Attribute(key=key, str=value)
    if isinstance(value, bytes):
        return pb.Operator.Attribute(key=key, bytes_=value)
    if isinstance(value, bool):
        # NOTE: Check this before int (isinstance(False, int) is also True)
        return pb.Operator.Attribute(key=key, boolean=value)
    if isinstance(value, int):
        return pb.Operator.Attribute(key=key, integer_64=value)
    if isinstance(value, float):
        return pb.Operator.Attribute(key=key, float_64=value)
    # list of strings
    if isinstance(value, list) and all(isinstance(val, str) for val in value):
        return pb.Operator.Attribute(
            key=key, list_str=pb.Operator.Attribute.ListString(values=value)
        )
    # map<str, str>
    if (
        isinstance(value, Mapping)
        and all(isinstance(key, str) for key in value.keys())
        and all(isinstance(val, str) for val in value.values())
    ):
        return pb.Operator.Attribute(
            key=key, map_str_str=pb.Operator.Attribute.MapStrStr(values=value)
        )
    # list of dtype
    if isinstance(value, list) and all(isinstance(val, DType) for val in value):
        return pb.Operator.Attribute(
            key=key,
            list_dtype=pb.Operator.Attribute.ListDType(
                values=[_serialize_dtype(x) for x in value]
            ),
        )
    # list of index keys
    if isinstance(value, list) and all(isinstance(val, tuple) for val in value):
        return pb.Operator.Attribute(
            key=key,
            list_index_keys=pb.Operator.Attribute.ListIndexKeys(
                values=[_serialize_index_key(x) for x in value]
            ),
        )

    raise ValueError(
        f"Non supported type {type(value)} for attribute {key}={value}"
    )


def _attribute_from_proto(src: pb.Operator.Attribute) -> base.AttributeType:
    if src.HasField("integer_64"):
        return src.integer_64
    if src.HasField("str"):
        return src.str
    if src.HasField("bytes_"):
        return src.bytes_
    if src.HasField("float_64"):
        return src.float_64
    if src.HasField("list_str"):
        return list(src.list_str.values)
    if src.HasField("boolean"):
        return bool(src.boolean)
    if src.HasField("map_str_str"):
        return dict(src.map_str_str.values)
    if src.HasField("list_dtype"):
        return [_unserialize_dtype(x) for x in src.list_dtype.values]
    if src.HasField("list_index_keys"):
        return [_unserialize_index_key(x) for x in src.list_index_keys.values]
    raise ValueError(f"Non supported proto attribute {src}")


def _serialize_io_signature(key: str, node: EventSetNode):
    return pb.IOSignature(
        key=key,
        node_id=_identifier(node),
    )
