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

import numpy as np
from typing import Dict, Tuple, Sequence
import apache_beam as beam

from temporian.core.operators.drop_index import (
    DropIndexOperator as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.beam.operators.base import (
    BeamOperatorImplementation,
)
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
    BeamIndexKey,
)


class DropIndexBeamImplementation(BeamOperatorImplementation):
    def call(self, input: BeamEventSet) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        src_index_names = self.operator.inputs["input"].schema.index_names()
        # Idx in src_index_names of the indexes to keep in the output.
        final_index_idxs = [
            idx
            for idx, name in enumerate(src_index_names)
            if name not in self.operator.indexes
        ]
        # Idx in src_index_names of the indexes to remove in the output.
        final_nonindex_idxs = [
            idx
            for idx, name in enumerate(src_index_names)
            if name in self.operator.indexes
        ]

        def build_new_index(item: FeatureItem) -> BeamIndexKey:
            """Creates the new index without the dropped items."""
            src_indexes, _ = item
            return tuple((src_indexes[i] for i in final_index_idxs))

        def merge_events(
            group: Tuple[BeamIndexKey, Sequence[FeatureItem]],
        ) -> FeatureItem:
            """Merges together events in the same output index."""
            new_indexes, items = group
            all_timestamps = []
            all_values = []
            values_are_none = False
            for item in items:
                _, (item_timestamps, item_values) = item
                all_timestamps.append(item_timestamps)
                all_values.append(item_values)
                if item_values is None:
                    values_are_none = True

            aggregated_timestamps = np.concatenate(all_timestamps)
            sorted_idxs = np.argsort(aggregated_timestamps, kind="mergesort")
            new_timestamps = aggregated_timestamps[sorted_idxs]
            if values_are_none:
                new_features = None
            else:
                new_features = np.concatenate(all_values)[sorted_idxs]
            return new_indexes, (new_timestamps, new_features)

        def src_index_to_feature(
            any_group: Tuple[BeamIndexKey, Sequence[FeatureItem]],
            final_nonindex_idx: int,
        ) -> FeatureItem:
            """Create a feature with the dropped index."""
            new_indexes, items = any_group
            all_timestamps = []
            all_values = []
            for item in items:
                old_index, (item_timestamps, _) = item
                item_values = np.full(
                    shape=len(item_timestamps),
                    fill_value=old_index[final_nonindex_idx],
                )
                all_timestamps.append(item_timestamps)
                all_values.append(item_values)
            aggregated_timestamps = np.concatenate(all_timestamps)
            sorted_idxs = np.argsort(aggregated_timestamps, kind="mergesort")
            new_timestamps = aggregated_timestamps[sorted_idxs]
            new_features = np.concatenate(all_values)[sorted_idxs]
            return new_indexes, (new_timestamps, new_features)

        def feature_less_event(
            any_group: Tuple[BeamIndexKey, Sequence[FeatureItem]],
        ) -> FeatureItem:
            """Create an event without feature."""
            new_indexes, items = any_group
            all_timestamps = []
            for item in items:
                _, (item_timestamps, _) = item
                all_timestamps.append(item_timestamps)
            aggregated_timestamps = np.concatenate(all_timestamps)
            sorted_idxs = np.argsort(aggregated_timestamps, kind="mergesort")
            new_timestamps = aggregated_timestamps[sorted_idxs]
            return new_indexes, (new_timestamps, None)

        group_by_new_index = tuple(
            item
            | f"Gen new index on feature #{feature_idx} {self.operator}"
            >> beam.GroupBy(build_new_index)
            for feature_idx, item in enumerate(input)
        )
        assert len(group_by_new_index) >= 1

        has_input_features = len(self.operator.inputs["input"].features) > 0

        if not has_input_features:
            existing_features_output = ()
            assert len(group_by_new_index) == 1
        else:
            existing_features_output = tuple(
                [
                    item
                    | f"Merge event on feature #{feature_idx} {self.operator}"
                    >> beam.Map(merge_events)
                    for feature_idx, item in enumerate(group_by_new_index)
                ]
            )

        if self.operator.keep:
            new_features_output = tuple(
                group_by_new_index[0]
                | f"Create index feature for src index #{src_index_idx} {self.operator}"
                >> beam.Map(src_index_to_feature, src_index_idx)
                for src_index_idx in final_nonindex_idxs
            )
        else:
            if has_input_features:
                new_features_output = tuple()
            else:
                new_features_output = tuple(
                    [
                        group_by_new_index[0]
                        | f"Create feature-less event {self.operator}"
                        >> beam.Map(feature_less_event)
                    ]
                )

        return {"output": existing_features_output + new_features_output}


implementation_lib.register_operator_implementation(
    CurrentOperator, DropIndexBeamImplementation
)
