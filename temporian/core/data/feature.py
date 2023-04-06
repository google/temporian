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

"""A feature."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from temporian.core.data.sampling import Sampling
from temporian.core.data import dtype as dtype_lib

if TYPE_CHECKING:
    from temporian.core.operators.base import Operator


class Feature(object):
    def __init__(
        self,
        name: str,
        dtype: dtype_lib.DType = dtype_lib.FLOAT32,
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

        if dtype not in dtype_lib.ALL_TYPES:
            raise ValueError(
                f"Invalid dtype feature constructor. Got {dtype}. "
                f"Expecting one of {dtype_lib.ALL_TYPES} instead."
            )

        self._name = name
        self._sampling = sampling
        self._dtype = dtype
        self._creator = creator

    def __repr__(self):
        return (
            f"name: {self.name}\n"
            f"  dtype: {self.dtype}\n"
            f"  sampling: {self.sampling}\n"
            f"  creator: {self.creator}\n"
            f"  id: {id(self)}"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> dtype_lib.DType:
        return self._dtype

    @property
    def sampling(self) -> Optional[Sampling]:
        return self._sampling

    @property
    def creator(self) -> Optional[Operator]:
        return self._creator

    @sampling.setter
    def sampling(self, sampling: Optional[Sampling]):
        self._sampling = sampling

    @creator.setter
    def creator(self, creator: Optional[Operator]):
        self._creator = creator
