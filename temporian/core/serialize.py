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

from typing import List, Set, Union, Any

from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base
from temporian.core.data import dtype as dtype_lib

from temporian.proto import core_pb2 as pb
from temporian.core import processor


def serialize(src: processor.Preprocessor) -> pb.Processor:
    """Serializes a processor into an equivalent protobuffer."""

    return pb.Processor(
        operators=[_serialize_operator(o) for o in src.operators()],
        events=[_serialize_event(o) for o in src.events()],
        features=[_serialize_feature(o) for o in src.features()],
        samplings=[_serialize_sampling(o) for o in src.samplings()],
    )


def unserialize(src: pb.Processor) -> processor.Preprocessor:
    """Unserializes a protobuffer into a processor."""

    pass


def _identifier(item: Any) -> str:
    return str(id(item))


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


def _serialize_event(src: Event) -> pb.Event:
    return pb.Event(
        id=_identifier(src),
        sampling_id=_identifier(src.sampling()),
        feature_ids=[_identifier(f) for f in src.features()],
    )


def _serialize_feature(src: Feature) -> pb.Feature:
    return pb.Feature(
        id=_identifier(src),
        type=_type_to_proto(src.dtype()),
        sampling_id=_identifier(src.sampling()),
        creator_event_id=_identifier(src.creator()),
    )


def _serialize_sampling(src: Sampling) -> pb.Sampling:
    return pb.Sampling(
        id=_identifier(src),
        index=src.index(),
        creator_event_id=_identifier(src.creator),
    )


def _type_to_proto(dtype) -> pb.Feature.Type:
    if dtype == dtype_lib.FLOAT:
        return pb.Feature.Type.FLOAT
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
