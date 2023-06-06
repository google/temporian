from collections import defaultdict
from typing import Dict, List

import numpy as np

from temporian.core.operators.drop_index import DropIndexOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet, IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation


class DstIndexGroup:
    def __init__(self) -> None:
        self.timestamps: List[np.ndarray] = []
        self.features: List[List[np.ndarray]] = []

    def __repr__(self):
        return f"timestamps:{self.timestamps}, features:{self.features}>"


class DropIndexNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: DropIndexOperator) -> None:
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, DropIndexOperator)

        src_index_names = input.schema.index_names()

        # Idx in src_index_names of the indexes to keep in the output.
        final_index_idxs = [
            idx
            for idx, f_name in enumerate(src_index_names)
            if f_name not in self.operator.index_to_drop
        ]
        # Idx in src_index_names of the indexes to remove in the output.
        final_nonindex_idxs = [
            idx
            for idx, f_name in enumerate(src_index_names)
            if f_name in self.operator.index_to_drop
        ]
        # Non-aggregated (i.e., in separate containers) event data indexed by
        # the destination index.
        dst_index_groups: Dict[tuple, DstIndexGroup] = defaultdict(
            DstIndexGroup
        )
        # Compute "dst_index_groups".
        for src_index_key, src_index_data in input.data.items():
            new_features_data = []
            if self.operator.keep:
                # Convert the dropped indexes into features
                num_timestamps = len(src_index_data.timestamps)
                for idx in final_nonindex_idxs:
                    index_value = src_index_key[idx]
                    new_feature_data = np.full(
                        shape=num_timestamps, fill_value=index_value
                    )
                    new_features_data.append(new_feature_data)

            dst_index_key = tuple((src_index_key[i] for i in final_index_idxs))
            dst_index_group = dst_index_groups[dst_index_key]
            dst_index_group.timestamps.append(src_index_data.timestamps)

            # Note: The new features are added after the existing features.
            dst_index_group.features.append(
                src_index_data.features + new_features_data
            )

        # Aggredates the data
        #
        # TODO: this is merging sorted arrays, we should later improve this code
        # by avoiding the full sort
        dst_evset: Dict[tuple, IndexData] = {}
        output_schema = self.output_schema("output")
        num_output_features = len(self.operator.output_feature_schemas)
        for dst_index_key, group in dst_index_groups.items():
            # Append together all the timestamps.
            aggregated_timestamps = np.concatenate(group.timestamps)

            # Sort the timestamps.
            sorted_idxs = np.argsort(aggregated_timestamps, kind="mergesort")

            # Append together and sort (according to the timestamps) all the feature values.
            aggregated_features = [
                np.concatenate([f[idx] for f in group.features])[sorted_idxs]
                for idx in range(num_output_features)
            ]

            dst_evset[dst_index_key] = IndexData(
                features=aggregated_features,
                timestamps=aggregated_timestamps[sorted_idxs],
                schema=output_schema,
            )

        return {"output": EventSet(data=dst_evset, schema=output_schema)}


implementation_lib.register_operator_implementation(
    DropIndexOperator, DropIndexNumpyImplementation
)
