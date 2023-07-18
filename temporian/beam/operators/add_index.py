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


from typing import Dict, List, Tuple, Optional, Iterable

from collections import defaultdict
import apache_beam as beam
import numpy as np

from temporian.core.operators.add_index import (
    AddIndexOperator as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.beam.operators.base import BeamOperatorImplementation
from temporian.beam.io import (
    IndexValue,
    PEventSet,
    BeamIndexOrFeature,
)

ExtractedIndex = Tuple[
    Tuple[BeamIndexOrFeature, ...], Tuple[int, Optional[np.ndarray]]
]


class AddIndexBeamImplementation(BeamOperatorImplementation):
    def call(self, input: PEventSet) -> Dict[str, PEventSet]:
        """AddIndex implementation.

        Example:
            pipe
                # Two features, one index
                (20, 0), ( (100, 101), (11, 13))
                (20, 1), ( (100, 101), (12, 14))

            # Adding feature #0 to the index
            indexes: [0]

            new_index_idxs: [0]
            kept_feature_idxs: [1]

            Output
                (20, 0, 11), ( (100,), (12))
                (20, 0, 13), ( (101,), (14))
        """
        assert isinstance(self.operator, CurrentOperator)

        # Idx of input features added to index.
        src_feature_names = self.operator.inputs["input"].schema.feature_names()
        new_index_idxs = [
            src_feature_names.index(f_name) for f_name in self.operator.indexes
        ]

        # Idx of input features not added to index.
        kept_feature_idxs = [
            idx
            for idx, f_name in enumerate(src_feature_names)
            if f_name not in self.operator.indexes
        ]

        # Broadcast the data of each new index to each remaining feature.
        extract_index = (
            input
            | f"Broadcast index to feature {self.operator}"
            >> beam.ParDo(
                _broadcast_index_to_feature, new_index_idxs, kept_feature_idxs
            )
        )

        # Join the new index and the remaining feature data, and compute the
        # new event set.
        output = (
            (input, extract_index)
            | f"Join feature and new index {self.operator}"
            >> beam.CoGroupByKey()
            | f"Reindex {self.operator}" >> beam.ParDo(_add_index)
        )

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, AddIndexBeamImplementation
)


def _broadcast_index_to_feature(
    pipe: IndexValue, new_index_idxs: List[int], kept_feature_idxs: List[int]
) -> ExtractedIndex:
    """Broadcast of each index-converted-feature to all other features.

    Input
        index + feature_index, (timestamps, feature_values)
    Output
        index + other_feature_index, (local_feature_index, feature_values)

    Note
        feature_index is a feature converted into a index.
        local_feature_index is the index idx (if the schema) of the newly added
            index (previously a feature).
    """

    indexes, (_, input_values) = pipe
    feature_idx = indexes[-1]

    if feature_idx not in new_index_idxs:
        # This is not the data of a new index.
        return

    local_idx = new_index_idxs.index(feature_idx)
    for kept_feature_idx in kept_feature_idxs:
        yield indexes[:-1] + (kept_feature_idx,), (local_idx, input_values)


def _add_index(
    items: Tuple[
        Tuple[BeamIndexOrFeature, ...],
        Tuple[
            Iterable[Tuple[np.ndarray, Optional[np.ndarray]]],
            Iterable[ExtractedIndex],
        ],
    ]
) -> IndexValue:
    """Adds the new index values to all remaining feature items."""

    indexes_and_feature_idx, (features, new_index) = items
    indexes = indexes_and_feature_idx[:-1]
    feature_idx = indexes_and_feature_idx[-1]

    # Extract the new index
    new_index = list(new_index)
    if len(new_index) == 0:
        # This feature is dropped
        return
    sorted(new_index, key=lambda x: x[0])

    # Extract the existing timestamps and feature values
    features = list(features)
    assert len(features) == 1
    timestamps, values = features[0]

    # The objective is now to combine the new index and the timestamp/feature
    # values. Note that different index items will be emitted as different items
    # in PEventSet.

    # Compute the example idxs for each unique index value.
    #
    # Note: This solution is very slow. This is the same used in the in-process
    # implementation.
    new_index_to_value_idxs = defaultdict(list)
    for event_idx, new_index in enumerate(zip(*[x[1] for x in new_index])):
        new_index = tuple([v.item() for v in new_index])
        new_index_to_value_idxs[new_index].append(event_idx)

    for new_index, example_idxs in new_index_to_value_idxs.items():
        # Note: The new index is added after the existing index items.
        dst_indexes = indexes + new_index + (feature_idx,)
        assert isinstance(dst_indexes, tuple)
        # This is the "PEventSet" format.
        yield dst_indexes, (timestamps[example_idxs], values[example_idxs])
