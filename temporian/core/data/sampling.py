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
from typing import List, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from temporian.core.operators.base import Operator


class Sampling(object):
    def __init__(
        self,
        index: List[str],
        creator: Optional[Operator] = None,
        is_unix_timestamp: bool = False,
    ):
        assert isinstance(index, list), f"Got {index}"

        self._index = index
        self._creator = creator
        self._is_unix_timestamp = is_unix_timestamp

    def __repr__(self):
        return f"Sampling<index:{self.index},id:{id(self)}>"

    @property
    def index(self) -> List[str]:
        return self._index

    @property
    def creator(self) -> Optional[Operator]:
        return self._creator

    @property
    def is_unix_timestamp(self) -> bool:
        return self._is_unix_timestamp

    @creator.setter
    def creator(self, creator: Optional[Operator]):
        self._creator = creator
