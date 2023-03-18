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

from typing import List, Optional, Any


class Sampling(object):
    def __init__(
        self,
        index: List[str],
        creator: Optional[Any] = None,
    ):
        assert isinstance(index, list), f"Got {index}"

        self._index: List[str] = index
        self._creator = creator

    def __repr__(self):
        return f"Sampling<index:{self._index},id:{id(self)}>"

    def index(self) -> List[str]:
        return self._index

    def creator(self):
        return self._creator

    def set_creator(self, creator):
        self._creator = creator
