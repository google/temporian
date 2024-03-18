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


from typing import Dict, List, Tuple, Iterable, Iterator

from collections import defaultdict
import apache_beam as beam
import numpy as np

from temporian.core.operators.add_index import (
    AddIndexOperator as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.beam.operators.base import BeamOperatorImplementation
from temporian.beam.typing import (
    BeamEventSet,
    BeamIndexKey,
    FeatureItem,
    FeatureItemValue,
    POS_FEATURE_VALUES,
)


class AddIndexBeamImplementation(BeamOperatorImplementation):
    def call(self, input: BeamEventSet) -> Dict[str, BeamEventSet]:
        """AddIndex implementation.

        Example:
            pipe
                # Two features, one index
                Feature #0
                    (20,), ((100, 101), (11, 13))
                Feature #1
                    (20,), ((100, 101), (12, 14))

            # Adding feature #1 to the index
            indexes: [1]

            new_index_idxs: [1]
            kept_feature_idxs: [0]

            Output
                Feature #0
                    (20, 12), ((100,), (11))
                    (20, 14), ((101,), (13))
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

        # Tuple of index features
        index_pipes = []
        for new_index_idx in new_index_idxs:
            index_pipes.append(input[new_index_idx])
        index_pipes = tuple(index_pipes)

        output = []
        for kept_feature_idx in kept_feature_idxs:
            # A features + all the indexes
            single_feature_and_index_pipes = (
                input[kept_feature_idx],
            ) + index_pipes

            output.append(
                single_feature_and_index_pipes
                | f"Join feature #{kept_feature_idx} with index {self.operator}"
                >> beam.CoGroupByKey()
                | f"Reindex feature #{kept_feature_idx} {self.operator}"
                >> beam.FlatMap(_add_index_to_feature)
            )

        return {"output": tuple(output)}


implementation_lib.register_operator_implementation(
    CurrentOperator, AddIndexBeamImplementation
)


def _add_index_to_feature(
    items: Tuple[
        BeamIndexKey,
        Tuple[Iterable[FeatureItemValue], ...],
    ],
) -> Iterator[FeatureItem]:
    """Adds the new index values to all remaining feature items."""

    old_index, mess = items

    # Note: "mess" contains exactly one value in each "Iterable".
    feature = next(iter(mess[0]))
    indexes_values = [next(iter(item))[POS_FEATURE_VALUES] for item in mess[1:]]

    timestamps, feature_values = feature
    assert feature_values is not None

    # Compute the example idxs for each unique index value.
    #
    # Note: This solution is very slow. This is the same used in the in-process
    # implementation.
    new_index_to_value_idxs = defaultdict(list)
    for event_idx, new_index in enumerate(zip(*indexes_values)):
        new_index = tuple([v.item() for v in new_index])
        new_index_to_value_idxs[new_index].append(event_idx)

    for new_index, example_idxs in new_index_to_value_idxs.items():
        # Note: The new index is added after the existing index items.
        dst_indexes = old_index + new_index
        assert isinstance(dst_indexes, tuple)
        # This is the "BeamEventSet" format.
        yield dst_indexes, (
            timestamps[example_idxs],
            feature_values[example_idxs],
        )
