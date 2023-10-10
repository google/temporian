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

"""Types used throughout the API."""

# NOTE: this module should only contain type imports to be public under tp.types
# Don't define classes or types here nor import stuff from here, will probably
# result in circular imports.

from temporian.core.typing import (
    EventSetCollection,
    EventSetNodeCollection,
    EventSetOrNode,
    IndexKey,
    IndexKeyItem,
    IndexKeyList,
    NodeToEventSetMapping,
    WindowLength,
)
from temporian.core.dataclasses import MapExtras
from temporian.core.operators.map import MapFunction

# Add all the types that are part of the public API and should be shown in docs.
__all__ = [
    "EventSetCollection",
    "EventSetNodeCollection",
    "EventSetOrNode",
    "IndexKey",
    "IndexKeyItem",
    "IndexKeyList",
    "MapExtras",
    "NodeToEventSetMapping",
    "WindowLength",
    "MapFunction",
]
