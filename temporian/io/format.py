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

"""Format choices definitions and documentation for I/O functions."""

from enum import Enum
from typing import Literal, Union


# NOTE: GroupedOrSingleEventSetFormatChoices' values in sync with enum (below)
class GroupedOrSingleEventSetFormat(str, Enum):
    """Format choices for converting dictionaries to and from EventSets.

    The GROUPED_BY_INDEX value is generally recommended as it is more efficient
    than SINGLE_EVENTS.
    """

    GROUPED_BY_INDEX = "grouped_by_index"
    """Events in the same index are grouped together in a dictionary mapping
    index value, features and timestamps to actual values.

    In this dictionary, the features and timestamp keys are mapped to numpy
    arrays containing one value per event, and index keys are mapped to single
    value python primitives (e.g., `int`, `float`, `bytes`).

    The dtype of each numpy array matches the Temporian dtype. For instance, a
    Temporian feature with `dtype=tp.int32` is stored as a numpy array with
    `dtype=np.int32`.

    For example, an EventSet with 3 events and the following Schema:

    ```
    features=[("f1", tp.int64), ("f2", tp.str_)]
    indexes=[("i1", tp.int64), ("i2", tp.str_)]
    ```

    would be represented as the following dictionary:

    ```
    {
    "timestamp": np.array([100.0, 101.0, 102.0], np.float64),
    "f1": np.array([1, 2, 3], np.int64),
    "f2": np.array([b"a", b"b", b"c"], np.bytes_),
    "i1": 10,
    "i2": b"x",
    }
    ```
    """

    SINGLE_EVENTS = "single_events"
    """Each event is represented as an individual dictionary of keys to unique
    values. Each index value, feature and timestamp is represented by an
    independent dictionary.

    For example, the same EventSet with 3 events and the following Schema:

    ```
    features=[("f1", tp.int64), ("f2", tp.str_)]
    indexes=[("i1", tp.int64), ("i2", tp.str_)]
    ```

    would be represented as the following dictionaries:

    ```
    {"timestamp": 100.0, "f1": 1, "f2": b"a", "i1": 10, "i2": b"x"}
    {"timestamp": 101.0, "f1": 2, "f2": b"b", "i1": 10, "i2": b"x"}
    {"timestamp": 102.0, "f1": 3, "f2": b"c", "i1": 10, "i2": b"x"}
    ```
    """


GroupedOrSingleEventSetFormatChoices = Union[
    GroupedOrSingleEventSetFormat, Literal["grouped_by_index", "single_events"]
]


DictEventSetFormat = GroupedOrSingleEventSetFormat
"""See [GroupedOrSingleEventSetFormat][temporian.io.format.GroupedOrSingleEventSetFormat]."""
DictEventSetFormatChoices = GroupedOrSingleEventSetFormatChoices

TFRecordEventSetFormat = GroupedOrSingleEventSetFormat
"""See [GroupedOrSingleEventSetFormat][temporian.io.format.GroupedOrSingleEventSetFormat]."""
TFRecordEventSetFormatChoices = GroupedOrSingleEventSetFormatChoices
