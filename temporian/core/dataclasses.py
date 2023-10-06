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

"""Data class definitions."""

from dataclasses import dataclass

from temporian.core.typing import IndexKey


@dataclass
class MapExtras:
    """Object containing information about a value's position in an
    [`EventSet`][temporian.EventSet].

    Attributes:
        index_key: The index the value belongs to.
        timestamp: The timestamp of the value's event.
        feature_name: The name of the feature the value belongs to.
    """

    index_key: IndexKey
    timestamp: float
    feature_name: str
