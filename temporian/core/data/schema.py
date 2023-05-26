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

from typing import List, Tuple, TYPE_CHECKING, Union

from dataclasses import dataclass
from temporian.core.data.dtype import DType, IndexDType

if TYPE_CHECKING:
    from temporian.core.operators.base import Operator


@dataclass
class FeatureSchema:
    name: str
    dtype: DType


@dataclass
class IndexSchema:
    name: str
    dtype: IndexDType


class Schema:
    """A schema defines the type of data in an event set.

    A schema does not contain any actual data.
    A schema can be shared by multiple nodes.

    Attributes:
        features: List of feature names and types.
        indexes: List of index names ans types.
        is_unix_timestamp: Whether values correspond to Unix timestamps.
    """

    def __init__(
        self,
        features: List[Union[FeatureSchema, Tuple[str, DType]]],
        indexes: List[Union[IndexSchema, Tuple[str, IndexDType]]],
        is_unix_timestamp: bool,
    ):
        def normalize_feature(x):
            if isinstance(x, FeatureSchema):
                return x
            assert len(x) == 2
            return FeatureSchema(x[0], x[1])

        def normalize_index(x):
            if isinstance(x, IndexSchema):
                return x
            assert len(x) == 2
            return FeatureSchema(x[0], x[1])

        self._features = map(normalize_feature, features)
        self._indexes = map(normalize_index, indexes)
        self._is_unix_timestamp = is_unix_timestamp

    @property
    def features(self) -> List[FeatureSchema]:
        return self._features

    @property
    def indexes(self) -> List[IndexSchema]:
        return self._indexes

    @property
    def is_unix_timestamp(self) -> bool:
        return self._is_unix_timestamp

    @property
    def feature_names(self) -> List[str]:
        return [feature.name for feature in self._features]

    @property
    def index_names(self) -> List[str]:
        return [index.name for index in self._indexes]

    @property
    def feature_dtypes(self) -> List[DType]:
        return [feature.dtype for feature in self._features]

    @property
    def index_dtypes(self) -> List[DType]:
        return [index.dtype for index in self._indexes]
