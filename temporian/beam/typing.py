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

from typing import Tuple, Union, Optional, List, Any

import numpy.typing as npt
import numpy as np
import apache_beam as beam
from temporian.core.typing import IndexKeyItem

# When applicable, follow the same naming convention as temporian/core/typing.py
#
# The Beam computation relies eavily on tuples instead of dataclasses or named
# tuples as it seems tuples are the most efficient solution.
#
# Individual features of a same event set are stored separately.
# An important implication is that timestamps are effectively repeated for
# each feature.

# Temporian index in Beam.
BeamIndexKeyItem = Union[int, bytes]
BeamIndexKey = Tuple[BeamIndexKeyItem, ...]

# Timestamp values
TimestampsDType = np.float64
TimestampValues = npt.NDArray[TimestampsDType]
TimestampsPyType = float

# StructuredRow is a single event / row during the conversion from dict of
# key/value to internal the Temporian beam format for EventSets. In a
# StructuredRow, index and features are indexed by integer instead of string
# keys.
SingleFeatureValue = Any  # np.generic
StructuredRowValue = Tuple[TimestampsPyType, Tuple[SingleFeatureValue, ...]]
StructuredRow = Tuple[BeamIndexKey, StructuredRowValue]

# A BeamFeatureAndTimestamps contains the values of a features + timestamps for
# an index. This is the internal representation used during the op computation.
#
# If no feature is available (i.e. event-set without features), the
# "FeatureValues" is set to None and the FeatureIdx is set to -1.
FeatureValues = np.ndarray
FeatureIdx = int
PosFeatureIdx = 0
PosTimestampValues = 1
PosFeatureValues = 2
BeamFeatureAndTimestampsValue = Tuple[
    FeatureIdx, TimestampValues, Optional[FeatureValues]
]
BeamFeatureAndTimestamps = Tuple[BeamIndexKey, BeamFeatureAndTimestampsValue]

# From the point of view of the user, a BeamEventSet play the same role as
# and EventSet with Temporian in-process.
BeamEventSet = tuple[beam.PCollection[BeamFeatureAndTimestamps]]
