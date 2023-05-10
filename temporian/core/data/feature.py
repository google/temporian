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

"""Feature class definition."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from temporian.core.data.sampling import Sampling
from temporian.core.data.dtype import DType

if TYPE_CHECKING:
    from temporian.core.operators.base import Operator


class Feature(object):
    """Feature values for a certain sampling.

    A feature can be thought of as a list of univariate series, where each item
    in the list corresponds to an index value in the feature's sampling and each
    value in that item corresponds to a timestamp in it.

    A feature holds no actual data, but rather describes the structure (or
    "schema") of data during evaluation.

    Attributes:
        creator: Operator that created this feature. Can be None if it wasn't
            created by an operator.
        dtype: Feature's data type.
        name: Name of the feature.
        sampling: Feature's sampling.
    """

    def __init__(
        self,
        name: str,
        dtype: DType = DType.FLOAT32,
        sampling: Optional[Sampling] = None,
        creator: Optional[Operator] = None,
    ):
        # TODO: Find a simple, efficient and consistant way to check the type
        # of arguments in the API.
        assert isinstance(
            name, str
        ), f"`name` must be a string. Got name={name} instead."
        assert sampling is None or isinstance(sampling, Sampling), (
            "`sampling` must be None or a Sampling. Got"
            f" sampling={sampling} instead."
        )

        if not isinstance(dtype, DType):
            raise ValueError(
                f"Invalid dtype feature constructor. Got {dtype}. "
                f"Expecting one of {DType} instead."
            )

        self._name = name
        self._sampling = sampling
        self._dtype = dtype
        self._creator = creator

    def to_string(self, include_sampling: bool = True) -> str:
        v = f"name: {self.name}\n  dtype: {self.dtype}\n"
        if include_sampling:
            v += f"  sampling: {self.sampling}\n"
        v += f"  creator: {self.creator}\n  id: {id(self)}"
        return v

    def __repr__(self):
        return self.to_string()

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def sampling(self) -> Optional[Sampling]:
        return self._sampling

    @property
    def creator(self) -> Optional[Operator]:
        return self._creator

    # TODO: Remove. A feature is constant after its creation.
    @sampling.setter
    def sampling(self, sampling: Optional[Sampling]):
        self._sampling = sampling

    # TODO: Remove. A feature is constant after its creation.
    @creator.setter
    def creator(self, creator: Optional[Operator]):
        self._creator = creator
