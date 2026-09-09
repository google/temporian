from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from temporian.core.operators.add_index import AddIndexOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet, IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation


def _compute_groups(
    index_features: List[np.ndarray],
) -> Tuple[list, np.ndarray, np.ndarray]:
    """Groups row indices by the combined values of index_features.

    Returns:
        group_keys: list of tuples, one per unique group.
        row_idxs: flat int64 array of row indices ordered by group.
        group_begin_idx: int64 array of length len(group_keys)+1 with the
            start offset of each group in row_idxs.
    """
    if len(index_features) == 0 or index_features[0].shape[0] == 0:
        return [], np.array([], dtype=np.int64), np.array([0], dtype=np.int64)

    num_rows = index_features[0].shape[0]

    # Build a dict mapping group_key -> list of row indices.
    groups: dict = defaultdict(list)
    for row_idx in range(num_rows):
        key = tuple(
            int(f[row_idx]) if f.dtype.kind in ("i", "u")
            else bytes(f[row_idx])
            for f in index_features
        )
        groups[key].append(row_idx)

    group_keys = []
    all_row_idxs = []
    begin_offsets = [0]
    for key, rows in groups.items():
        group_keys.append(key)
        all_row_idxs.extend(rows)
        begin_offsets.append(len(all_row_idxs))

    row_idxs = np.array(all_row_idxs, dtype=np.int64)
    group_begin_idx = np.array(begin_offsets, dtype=np.int64)
    return group_keys, row_idxs, group_begin_idx


class AddIndexNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the set index operator."""

    def __init__(self, operator: AddIndexOperator) -> None:
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, AddIndexOperator)
        output_node = self.operator.outputs["output"]

        # Idx of input features added to index.
        src_feature_names = input.schema.feature_names()
        new_index_idxs = [
            src_feature_names.index(f_name) for f_name in self.operator.indexes
        ]

        # Idx of input features not added to index.
        kept_feature_idxs = [
            idx
            for idx, f_name in enumerate(src_feature_names)
            if f_name not in self.operator.indexes
        ]

        dst_data = {}
        for src_index, src_data in input.data.items():
            index_features = [src_data.features[i] for i in new_index_idxs]
            group_keys, row_idxs, group_begin_idx = _compute_groups(
                index_features
            )

            for group_idx, group_key in enumerate(group_keys):
                dst_index = src_index + group_key
                assert isinstance(dst_index, tuple)

                example_idxs = row_idxs[
                    group_begin_idx[group_idx] : group_begin_idx[group_idx + 1]
                ]
                dst_data[dst_index] = IndexData(
                    features=[
                        src_data.features[f_idx][example_idxs]
                        for f_idx in kept_feature_idxs
                    ],
                    timestamps=src_data.timestamps[example_idxs],
                    schema=output_node.schema,
                )

        return {
            "output": EventSet(
                data=dst_data,
                schema=output_node.schema,
            )
        }


implementation_lib.register_operator_implementation(
    AddIndexOperator, AddIndexNumpyImplementation
)
