from collections import defaultdict
from typing import Dict, List

import numpy as np

from temporian.core.operators.drop_index import DropIndexOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import DTYPE_REVERSE_MAPPING
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

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        index_to_drop = self.operator.index_to_drop
        keep = self.operator.keep
        dst_feature_names = self.operator.dst_feature_names()
        src_index_dtypes = event.index_dtypes()
        src_index_names = event.index_names

        # Idx in src_index_names of the indexes to keep in the output.
        final_index_idxs = [
            idx
            for idx, name in enumerate(src_index_names)
            if name not in index_to_drop
        ]
        # Idx in src_index_names of the indexes to remove in the output.
        final_nonindex_idxs = [
            idx
            for idx, name in enumerate(src_index_names)
            if name in index_to_drop
        ]
        # Non-aggregated (i.e., in separate containers) event data indexed by
        # the destination index.
        dst_index_groups: Dict[tuple, DstIndexGroup] = defaultdict(
            DstIndexGroup
        )
        # Compute "dst_index_groups".
        for src_index_key, src_index_data in event.iterindex():
            dst_index_key = tuple((src_index_key[i] for i in final_index_idxs))
            dst_index_group = dst_index_groups[dst_index_key]

            features = []
            if keep:
                # Convert the dropped indexes into features
                num_timestamps = len(src_index_data.timestamps)
                for idx in final_nonindex_idxs:
                    index_name = src_index_names[idx]
                    index_value = src_index_key[idx]

                    index_data = np.array(
                        [index_value] * num_timestamps,
                        dtype=DTYPE_REVERSE_MAPPING[
                            src_index_dtypes[index_name]
                        ],
                    )
                    features.append(index_data)

            dst_index_group.timestamps.append(src_index_data.timestamps)
            dst_index_group.features.append(
                features + [feature for feature in src_index_data.features]
            )

        # Aggredates the data
        #
        # TODO: this is merging sorted arrays, we should later improve this code
        # by avoiding the full sort
        dst_event_data = {}
        for dst_index_key, group in dst_index_groups.items():
            # Append together all the timestamps.
            local_dst_sampling_data = np.concatenate(group.timestamps)

            # Sort the timestamps.
            sorted_idxs = np.argsort(local_dst_sampling_data, kind="mergesort")

            # Append together and sort (according to the timestamps) all the feature values.
            local_dst_event_data = [
                np.concatenate(
                    [features[dst_feature_idx] for features in group.features]
                )[sorted_idxs]
                for dst_feature_idx in range(len(dst_feature_names))
            ]
            dst_event_data[dst_index_key] = IndexData(
                local_dst_event_data, local_dst_sampling_data[sorted_idxs]
            )

        return {
            "event": NumpyEvent(
                data=dst_event_data,
                feature_names=dst_feature_names,
                index_names=self.operator.dst_index_names(),
                is_unix_timestamp=event.is_unix_timestamp,
            )
        }


implementation_lib.register_operator_implementation(
    DropIndexOperator, DropIndexNumpyImplementation
)
