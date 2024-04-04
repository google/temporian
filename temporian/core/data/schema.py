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

"""Schema and related classes."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union

from temporian.core.data.dtype import DType, IndexDType
from google.protobuf import text_format


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    dtype: DType

    def __repr__(self) -> str:
        return f"({self.name!r}, {self.dtype})"


@dataclass(frozen=True)
class IndexSchema:
    name: str
    dtype: IndexDType

    def __repr__(self) -> str:
        return f"({self.name!r}, {self.dtype})"


class Schema:
    """A schema defines the type of data in an
    [`EventSetNode`][temporian.EventSetNode] or
    [`EventSet`][temporian.EventSet].

    A schema does not contain any actual data.

    A schema can be shared by multiple EventSetNodes.

    Attributes:
        features: List of feature names and types.
        indexes: List of index names and types.
        is_unix_timestamp: Whether values correspond to Unix timestamps.
    """

    def __init__(
        self,
        features: Union[List[FeatureSchema], List[Tuple[str, DType]]],
        indexes: Union[List[IndexSchema], List[Tuple[str, IndexDType]]] = [],
        is_unix_timestamp: bool = False,
    ):
        self._features = list(map(_normalize_feature, features))
        self._indexes = list(map(_normalize_index, indexes))
        self._is_unix_timestamp = is_unix_timestamp

    def __eq__(self, other):
        if not isinstance(other, Schema):
            return False
        return (
            self._features == other._features
            and self._indexes == other._indexes
            and self._is_unix_timestamp == other._is_unix_timestamp
        )

    @property
    def features(self) -> List[FeatureSchema]:
        return self._features

    @property
    def indexes(self) -> List[IndexSchema]:
        return self._indexes

    @property
    def is_unix_timestamp(self) -> bool:
        return self._is_unix_timestamp

    @is_unix_timestamp.setter
    def is_unix_timestamp(self, value):
        self._is_unix_timestamp = value

    def feature_names(self) -> List[str]:
        return [feature.name for feature in self._features]

    def index_names(self) -> List[str]:
        return [index.name for index in self._indexes]

    def feature_dtypes(self) -> List[DType]:
        return [feature.dtype for feature in self._features]

    def index_dtypes(self) -> List[IndexDType]:
        return [index.dtype for index in self._indexes]

    def feature_name_to_dtype(self) -> Dict[str, DType]:
        return {feature.name: feature.dtype for feature in self._features}

    def index_name_to_dtype(self) -> Dict[str, IndexDType]:
        return {index.name: index.dtype for index in self._indexes}

    def __repr__(self) -> str:
        r = (
            f"features: {self._features}\n"
            f"indexes: {self._indexes}\n"
            f"is_unix_timestamp: {self._is_unix_timestamp}\n"
        )
        return r

    def check_compatible_index(self, other: Schema):
        if self.indexes != other.indexes:
            raise ValueError(
                f"Arguments don't have the same index. {self.indexes} !="
                f" {other.indexes}"
            )

    def check_compatible_features(self, other: Schema, check_order: bool):
        if set(self.features) != set(other.features):
            raise ValueError(
                "Some features are different between inputs: "
                f"{set(self.features) ^ set(other.features)}."
            )
        if check_order and (self.features != other.features):
            raise ValueError(
                "Features should be in the same order, but got:"
                f"{self.feature_names} and {other.feature_names}."
            )

    def to_proto(self) -> "serialization.pb.Schema":
        """Converts the schema into a protobuf.

        Usage example:
            ```
            schema = tp.Schema(features=[("f1",int), (f2, float)])
            proto_schema = schema.to_proto()
            restored_schema = tp.Schema.from_proto(proto_schema)
            ```
        """
        from temporian.core import serialization

        return serialization._serialize_schema(self)

    def to_proto_file(self, path: str) -> None:
        """Save the schema to a file with text protobuf format.

        Usage example:
            ```
            schema = tp.Schema(features=[("f1",int), (f2, float)])
            path = "/tmp/my_schema.pbtxt"
            schema.to_proto_file(path)
            restored_schema = tp.Schema.from_proto_file(path)
            ```
        """
        proto = self.to_proto()
        with open(path, "wb") as f:
            f.write(text_format.MessageToBytes(proto))

    @classmethod
    def from_proto(cls, proto: "serialization.pb.Schema") -> "Schema":
        """Creates a schema from a protobuf."""

        from temporian.core import serialization

        return serialization._unserialize_schema(proto)

    @classmethod
    def from_proto_file(cls, path: str) -> "Schema":
        """Creates a schema from a file text protobuf."""

        from temporian.core import serialization

        with open(path, "rb") as f:
            proto = text_format.Parse(f.read(), serialization.pb.Schema())
        return Schema.from_proto(proto)


def _normalize_feature(x):
    if isinstance(x, FeatureSchema):
        return x
    assert len(x) == 2
    return FeatureSchema(x[0], x[1])


def _normalize_index(x):
    if isinstance(x, IndexSchema):
        return x
    assert len(x) == 2
    return IndexSchema(x[0], x[1])
