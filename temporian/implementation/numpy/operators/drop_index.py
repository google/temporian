from collections import defaultdict
from typing import Dict, List

import numpy as np

from temporian.core.operators.drop_index import DropIndexOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.feature import DTYPE_REVERSE_MAPPING
from temporian.implementation.numpy.operators.base import OperatorImplementation


class DstIndexGroup:
    def __init__(self, num_src_indexes: int):
        self.num_timestamps: int = 0
        self.timestamps: List[np.ndarray] = []
        self.features: List[List[np.ndarray]] = []

    def __repr__(self):
        return (
            f"DstIndexGroup<num_timestamps:{self.num_timestamps}, "
            f"timestamps:{self.timestamps}, "
            f"features:{self.features}>"
        )


class DropIndexNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: DropIndexOperator) -> None:
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        index_to_drop = self.operator.index_to_drop
        keep = self.operator.keep
        dst_feature_names = self.operator.dst_feature_names()
        src_index_dtypes = event.index_dtypes
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
            lambda: DstIndexGroup(len(src_index_names))
        )

        # Compute "dst_index_groups".
        for src_index_lvl, timestamps in event.sampling.data.items():
            dst_index_lvl = tuple((src_index_lvl[i] for i in final_index_idxs))
            dst_index_group = dst_index_groups[dst_index_lvl]
            num_timestamps = len(timestamps)

            features = []
            if keep:
                # Convert the dropped indexes into features
                for idx in final_nonindex_idxs:
                    index_name = src_index_names[idx]
                    index_value = src_index_lvl[idx]

                    index_data = np.array(
                        [index_value] * num_timestamps,
                        dtype=DTYPE_REVERSE_MAPPING[
                            src_index_dtypes[index_name]
                        ],
                    )
                    features.append(index_data)

            dst_index_group.num_timestamps += num_timestamps
            dst_index_group.timestamps.append(timestamps)
            dst_index_group.features.append(
                features + [f.data for f in event.data[src_index_lvl]]
            )

        # Aggredates the data
        #
        # TODO: this is merging sorted arrays, we should later improve this code
        # by avoiding the full sort
        dst_event_data = {}
        for dst_index_lvl, group in dst_index_groups.items():
            # Append together all the timestamps.
            local_dst_sampling_data = np.concatenate(group.timestamps)

            # Sort the timestamps.
            sorted_idxs = np.argsort(local_dst_sampling_data, kind="mergesort")

            # Append together and sort (according to the timestamps) all the feature values.
            local_dst_event_data = []
            for dst_feature_idx, dst_feature_name in enumerate(
                dst_feature_names
            ):
                raw_data = [f[dst_feature_idx] for f in group.features]
                local_dst_event_data.append(
                    data=np.concatenate(raw_data)[sorted_idxs],
                )
            dst_event_data[dst_index_lvl] = IndexData(
                local_dst_event_data, local_dst_sampling_data[sorted_idxs]
            )

        return {
            "event": NumpyEvent(
<<<<<<< HEAD
                data=dst_event_data,
                feature_names=dst_feature_names,
                index_names=self.operator.dst_feature_names(),
=======
                dst_data,
                feature_names=dst_feat_names,
                index_names=dst_index_names,
>>>>>>> de5a95a (Address initial PR #95 comments)
                is_unix_timestamp=event.is_unix_timestamp,
            )
        }


implementation_lib.register_operator_implementation(
    DropIndexOperator, DropIndexNumpyImplementation
)
