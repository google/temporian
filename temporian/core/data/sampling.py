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

from typing import Any, Dict, Optional, Union

from temporian.core.data import dtype as dtype_lib


IndexDtypes = Union[dtype_lib.INT32, dtype_lib.INT64, dtype_lib.STRING]


class Sampling(object):
    def __init__(
        self,
        index: Dict[str, IndexDtypes],
        creator: Optional[Any] = None,
        is_unix_timestamp: bool = False,
    ):
        assert isinstance(index, dict), f"Got {index}"

        self._index: Dict[str, IndexDtypes] = index
        self._creator = creator
        self._is_unix_timestamp = is_unix_timestamp

    def __repr__(self):
        return f"Sampling<index:{self._index},id:{id(self)}>"

    def index(self) -> Dict[str, IndexDtypes]:
        return self._index

    def creator(self):
        return self._creator

    def set_creator(self, creator):
        self._creator = creator

    def is_unix_timestamp(self) -> bool:
        return self._is_unix_timestamp
