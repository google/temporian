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

"""Serialization / unserialization of a processor."""

from google.protobuf import text_format

from typing import Set, Union, Any, Dict, Tuple

from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base
from temporian.core.data import dtype as dtype_lib
from temporian.core import operator_lib
from temporian.proto import core_pb2 as pb
from temporian.core import processor


def save(
    inputs: Dict[str, Event], outputs: Dict[str, Event], path: str
) -> None:
    """Saves the computation between "inputs" and "outputs" into a file.

    Usage example:
        a = t.place_holder(...)
        b = t.sma(a, window_length=7)
        t.save(inputs={"io_a": a}, outputs={"io_b": b}, path="processor.tem")

        inputs, outputs = t.load(path="processor.tem")
        print(t.evaluate(
            query=outputs["io_b"],
            input_data{inputs["io_a"]: pandas.DataFrame(...)}
        ))


    Args:
        inputs: The inputs events.
        outputs: The output events.
        path: The file path.
    """

    # TODO: Add support for compressed / binary serialization.

    p = processor.infer_processor(inputs=inputs, outputs=outputs)
    proto = serialize(p)
    with open(path, "w") as f:
        f.write(text_format.MessageToString(proto))


def load(path: str) -> Tuple[Dict[str, Event], Dict[str, Event]]:
    """Load a processor from a file.

    Args:
        path: File path.

    Returns:
        The inputs and outputs events.
    """

    with open(path, "r") as f:
        proto = text_format.Parse(f.read(), pb.Processor())
    p = unserialize(proto)

    return p.inputs(), p.outputs()


def serialize(src: processor.Preprocessor) -> pb.Processor:
    """Serializes a processor into an equivalent protobuffer."""

    return pb.Processor(
        operators=[_serialize_operator(o) for o in src.operators()],
        events=[_serialize_event(o) for o in src.events()],
        features=[
            _serialize_feature(o, src.input_features()) for o in src.features()
        ],
        samplings=[
            _serialize_sampling(o, src.input_samplings())
            for o in src.samplings()
        ],
        inputs=[_serialize_io_signature(k, e) for k, e in src.inputs().items()],
        outputs=[
            _serialize_io_signature(k, e) for k, e in src.outputs().items()
        ],
    )


def unserialize(src: pb.Processor) -> processor.Preprocessor:
    """Unserializes a protobuffer into a processor."""

    # Decode the components.
    # All the fieds except for the "creator" ones are set.
    samplings = {s.id: _unserialize_sampling(s) for s in src.samplings}
    features = {s.id: _unserialize_feature(s, samplings) for s in src.features}
    events = {
        s.id: _unserialize_event(s, samplings, features) for s in src.events
    }
    operators = {s.id: _unserialize_operator(s, events) for s in src.operators}

    # Set the creator fields.
    def get_creator(op_id: str) -> base.Operator:
        if op_id not in operators:
            raise ValueError(f"Non existing creator operator {op_id}")
        return operators[op_id]

    for src_feature in src.features:
        if src_feature.creator_event_id:
            features[src_feature.id].set_creator(
                get_creator(src_feature.creator_event_id)
            )
    for src_sampling in src.samplings:
        if src_sampling.creator_event_id:
            samplings[src_sampling.id].set_creator(
                get_creator(src_sampling.creator_event_id)
            )

    # Copy extracted items.
    p = processor.Preprocessor()
    for sampling in samplings.values():
        p.samplings().add(sampling)
    for event in events.values():
        p.events().add(event)
    for feature in features.values():
        p.features().add(feature)
    for operator in operators.values():
        p.operators().add(operator)

    # IO Signature
    def get_event(event_id: str) -> Event:
        if event_id not in events:
            raise ValueError(f"Non existing event {event_id}")
        return events[event_id]

    for item in src.inputs:
        p.inputs()[item.key] = get_event(item.event_id)

    for item in src.outputs:
        p.outputs()[item.key] = get_event(item.event_id)

    return p


def _identifier(item: Any) -> str:
    """Unique identifier about an object within a processor."""

    return str(id(item))


def all_identifier(collection: Any) -> Set[str]:
    """Builds the set of identifiers of a collections of events/features/..."""

    return {_identifier(x) for x in collection}


def _serialize_operator(src: base.Operator) -> pb.Operator:
    return pb.Operator(
        id=_identifier(src),
        operator_def_key=src.definition().key,
        inputs=[
            pb.Operator.EventArgument(key=k, event_id=_identifier(v))
            for k, v in src.inputs().items()
        ],
        outputs=[
            pb.Operator.EventArgument(key=k, event_id=_identifier(v))
            for k, v in src.outputs().items()
        ],
        attributes=[
            _attribute_to_proto(k, v) for k, v in src.attributes().items()
        ],
    )


def _unserialize_operator(
    src: pb.Operator, events: Dict[str, Event]
) -> base.Operator:
    operator_class = operator_lib.get_operator_class(src.operator_def_key)

    def gen_event(key):
        if key not in events:
            raise ValueError(f"Non existing event {key}")
        return events[key]

    input_args = {x.key: gen_event(x.event_id) for x in src.inputs}
    output_args = {x.key: gen_event(x.event_id) for x in src.outputs}
    attribute_args = {x.key: _attribute_from_proto(x) for x in src.attributes}

    # We construct the operator.
    op = operator_class(**input_args, **attribute_args)

    # Check that the operator signature matches the expected one.
    if op.inputs().keys() != input_args.keys():
        raise ValueError(
            f"Restoring the operator {src.operator_def_key} lead "
            "to an unexpected input signature. "
            f"Expected: {input_args.keys()} Effective: {op.inputs().keys()}"
        )

    if op.outputs().keys() != output_args.keys():
        raise ValueError(
            f"Restoring the operator {src.operator_def_key} lead "
            "to an unexpected output signature. "
            f"Expected: {output_args.keys()} Effective: {op.outputs().keys()}"
        )

    if op.attributes().keys() != attribute_args.keys():
        raise ValueError(
            f"Restoring the operator {src.operator_def_key} lead to an"
            " unexpected attributes signature. Expected:"
            f" {attribute_args.keys()} Effective: {op.attributes().keys()}"
        )
    # TODO: Deep check of equality of the inputs / outputs / attributes

    # Override the inputs / outputs / attributes
    op.set_inputs(input_args)
    op.set_outputs(output_args)
    op.set_attributes(attribute_args)
    return op


def _serialize_event(src: Event) -> pb.Event:
    return pb.Event(
        id=_identifier(src),
        sampling_id=_identifier(src.sampling()),
        feature_ids=[_identifier(f) for f in src.features()],
    )


def _unserialize_event(
    src: pb.Event, samplings: Dict[str, Sampling], features: Dict[str, Feature]
) -> Event:
    if src.sampling_id not in samplings:
        raise ValueError(f"Non existing sampling {src.sampling_id}")

    def get_feature(key):
        if key not in features:
            raise ValueError(f"Non existing feature {key}")
        return features[key]

    return Event(
        sampling=samplings[src.sampling_id],
        features=[get_feature(f) for f in src.feature_ids],
    )


def _serialize_feature(
    src: Feature, input_features: Set[Feature]
) -> pb.Feature:
    return pb.Feature(
        id=_identifier(src),
        name=src.name(),
        dtype=_type_to_proto(src.dtype()),
        sampling_id=_identifier(src.sampling()),
        creator_event_id=(
            _identifier(src.creator()) if src not in input_features else None
        ),
    )


def _unserialize_feature(
    src: pb.Feature, samplings: Dict[str, Sampling]
) -> Feature:
    if src.sampling_id not in samplings:
        raise ValueError(f"Non existing sampling {src.sampling_id}")

    return Feature(
        name=src.name,
        dtype=_type_from_proto(src.dtype),
        sampling=samplings[src.sampling_id],
        creator=None,
    )


def _serialize_sampling(
    src: Sampling, input_samplings: Set[Feature]
) -> pb.Sampling:
    return pb.Sampling(
        id=_identifier(src),
        index=src.index(),
        creator_event_id=(
            _identifier(src.creator) if src not in input_samplings else None
        ),
    )


def _unserialize_sampling(src: pb.Sampling) -> Sampling:
    return Sampling(index=src.index, creator=None)


def _type_to_proto(dtype) -> pb.Feature.DType:
    if dtype == dtype_lib.FLOAT:
        return pb.Feature.DType.FLOAT
    else:
        raise ValueError(f"Non supported type {dtype}")


def _type_from_proto(dtype: pb.Feature.DType):
    if dtype == pb.Feature.DType.FLOAT:
        return dtype_lib.FLOAT
    else:
        raise ValueError(f"Non supported type {dtype}")


def _attribute_to_proto(
    key: str, value: Union[str, int]
) -> pb.Operator.Attribute:
    if isinstance(value, str):
        return pb.Operator.Attribute(key=key, str=value)
    elif isinstance(value, int):
        return pb.Operator.Attribute(key=key, integer_64=value)
    else:
        raise ValueError(
            f"Non supported type {type(value)} for attribute {key}={value}"
        )


def _attribute_from_proto(src: pb.Operator.Attribute) -> Union[str, int]:
    if src.HasField("integer_64"):
        return src.integer_64
    elif src.HasField("str"):
        return src.str
    else:
        raise ValueError(f"Non supported proto attribute {src}")


def _serialize_io_signature(key: str, event: Event):
    return pb.IOSignature(
        key=key,
        event_id=_identifier(event),
    )
