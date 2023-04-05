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

"""A sampling."""

from __future__ import annotations
from typing import Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union

from temporian.core.data import dtype as dtype_lib

if TYPE_CHECKING:
    from temporian.core.operators.base import Operator


IndexDtypes = Union[dtype_lib.INT32, dtype_lib.INT64, dtype_lib.STRING]


class IndexLevel(NamedTuple):
    name: str
    dtype: IndexDtypes


class Sampling(object):
    def __init__(
        self,
        index: List[Union[Tuple[str, IndexDtypes], IndexLevel]],
        creator: Optional[Operator] = None,
        is_unix_timestamp: bool = False,
    ):
        assert isinstance(index, list), f"Got {index}"
        if len(index) > 0:
            assert isinstance(index[0], tuple), f"Got {index[0]}"

        self._index = self._parse_index(index)
        self._creator = creator
        self._is_unix_timestamp = is_unix_timestamp

    def __repr__(self):
        return f"Sampling<index:{self.index},id:{id(self)}>"

    @property
    def index(self) -> List[IndexLevel]:
        return self._index

    @property
    def index_names(self) -> List[str]:
        return [index_level.name for index_level in self._index]

    @property
    def index_dtypes(self) -> Dict[str, IndexDtypes]:
        return {
            index_level.name: index_level.dtype for index_level in self._index
        }

    @property
    def creator(self) -> Optional[Operator]:
        return self._creator

    @property
    def is_unix_timestamp(self) -> bool:
        return self._is_unix_timestamp

    @creator.setter
    def creator(self, creator: Optional[Operator]):
        self._creator = creator

    def _parse_index(
        self, index: List[Union[Tuple[str, IndexDtypes], IndexLevel]]
    ) -> List[IndexLevel]:
        if all([isinstance(index_level, IndexLevel) for index_level in index]):
            return index

        return [
            IndexLevel(index_name, index_dtype)
            for index_name, index_dtype in index
        ]
