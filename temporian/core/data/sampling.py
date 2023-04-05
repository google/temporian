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

"""Sampling class definition."""

from typing import List, Optional, Any


class Sampling(object):
    """Lists of timestamps corresponding to unique values in an Index.

    A sampling consists of a (possibly multi-level) index, and a list of
    timestamps for each of the unique pairs of values present in it.

    A sampling holds no actual data, but rather describes the structure (or
    "schema") of data during evaluation.

    A sampling's values can correspond to actual Unix timestamps, but also to
    any arbitrary number that represents a time (e.g. the number of seconds that
    passed since an experiment's start time).

    Attributes:
        creator: the operator that created this sampling. Can be None if it
            wasn't created by an operator.
        index: the names of the columns that compose this sampling's index.
        is_unix_timestamp: whether values correspond to Unix timestamps.
    """

    def __init__(
        self,
        index: List[str],
        creator: Optional[Any] = None,
        is_unix_timestamp: bool = False,
    ):
        assert isinstance(index, list), f"Got {index}"

        self._index: List[str] = index
        self._creator = creator
        self._is_unix_timestamp = is_unix_timestamp

    def __repr__(self):
        return f"Sampling<index:{self._index},id:{id(self)}>"

    def index(self) -> List[str]:
        return self._index

    def creator(self):
        return self._creator

    def set_creator(self, creator):
        self._creator = creator

    def is_unix_timestamp(self) -> bool:
        return self._is_unix_timestamp
