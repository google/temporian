from collections import defaultdict
from typing import Dict, List, Any

import numpy as np

from temporian.core.operators.set_index import SetIndexOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import (
    DTYPE_REVERSE_MAPPING,
    DTYPE_MAPPING,
)
from temporian.implementation.numpy.operators.base import OperatorImplementation


class SetIndexNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the set index operator."""

    def __init__(self, operator: SetIndexOperator) -> None:
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        # get attributes
        feature_names = self.operator.attributes["feature_names"]
        append = self.operator.attributes["append"]

        if append:
            return {"output": _append_impl(input, feature_names)}

        return {"output": _set_impl(input, feature_names)}


def _append_impl(evset: EventSet, append_feat_names: List[str]) -> EventSet:
    """Appends the specified feature names to the index of the given event set.

    Args:
        evset: EventSet to append features to the index of.
        append_feat_names: List of feature names to append to the index.

    Returns:
        EventSet with the updated index.
    """
    # destination index names
    dst_idx_names = evset.index_names + append_feat_names

    # positions of features that are going to be part of the destination index
    dst_idx_pos = [
        evset.feature_names.index(append_feat_name)
        for append_feat_name in append_feat_names
    ]
    # positions of features that are being kept
    dst_feat_pos = [
        pos
        for pos, feat_name in enumerate(evset.feature_names)
        if feat_name not in append_feat_names
    ]
    # names of features that are being kept
    dst_feat_names = [evset.feature_names[pos] for pos in dst_feat_pos]

    # initialize destination event & sampling data
    dst_evset = {}
    for src_key, src_idx_data in evset.iterindex():
        # constructing the dict of unique tuples and their positions
        dst_suffs = defaultdict(list)
        for dst_suff_idx, dst_suff in enumerate(
            zip(*[src_idx_data.features[j] for j in dst_idx_pos])
        ):
            dst_suff = tuple(dst_suff)
            dst_suffs[dst_suff].append(dst_suff_idx)

        for dst_suff, dst_suff_idxs in dst_suffs.items():
            # create destination index
            dst_key = src_key + dst_suff

            dst_evset[dst_key] = IndexData(
                [
                    src_idx_data.features[feat_pos][dst_suff_idxs]
                    for feat_pos in dst_feat_pos
                ],
                # fill sampling data
                src_idx_data.timestamps[dst_suff_idxs],
            )

    # finally, sort according to timestamps. TODO: this is merging sorted
    # arrays, we should later improve this code by avoiding the full sort
    for dst_idx_data in dst_evset.values():
        sorted_idxs = np.argsort(dst_idx_data.timestamps, kind="mergesort")
        # features
        for i, feature in enumerate(dst_idx_data.features):
            dst_idx_data.features[i] = feature[sorted_idxs]

        # timestamp
        dst_idx_data.timestamps = dst_idx_data.timestamps[sorted_idxs]

    return EventSet(
        dst_evset,
        feature_names=dst_feat_names,
        index_names=dst_idx_names,
        is_unix_timestamp=evset.is_unix_timestamp,
    )


def _convert_to_index_type(value: Any) -> Any:
    """Converts a numpy value into an index compatible value type."""

    if value.dtype.type is np.str_:
        return str(value)

    if value not in DTYPE_MAPPING:
        raise ValueError(f"Non supported index type {type(value)}")

    return value


def _set_impl(evset: EventSet, set_feat_names: List[str]) -> EventSet:
    """Sets the specified feature names as the new index of the given event set.

    Args:
        evset: EventSet to set the index of.
        set_feat_names: List of feature names to set as the new index.

    Returns:
        Event with the updated index.
    """
    # positions of features that are going to be part of the destination index
    dst_idx_pos = [
        evset.feature_names.index(set_feat_name)
        for set_feat_name in set_feat_names
    ]
    # positions of features that are being kept
    dst_feat_pos = [
        pos
        for pos, feat_name in enumerate(evset.feature_names)
        if feat_name not in set_feat_names
    ]
    # names of features that are being kept
    dst_feat_names = [evset.feature_names[pos] for pos in dst_feat_pos]
    # intialize empty dict mapping destination index levels to timestamps &
    # features
    dst_idx_metadata = defaultdict(
        lambda: {
            "timestamps": [],
            "features": [[] for _ in dst_feat_pos],
        }
    )
    # loop over source index levels gathering destination index data
    for src_idx_data in evset.data.values():
        for dst_key_idx, dst_key in enumerate(
            zip(*[src_idx_data.features[j] for j in dst_idx_pos])
        ):
            dst_key = tuple([_convert_to_index_type(x) for x in dst_key])

            dst_idx_metadata[dst_key]["timestamps"].append(
                src_idx_data.timestamps[dst_key_idx]
            )
            for j, feat_pos in enumerate(dst_feat_pos):
                dst_idx_metadata[dst_key]["features"][j].append(
                    src_idx_data.features[feat_pos][dst_key_idx]
                )

    # create destination evset
    dst_evset = {
        dst_key: IndexData(
            [
                np.array(
                    metadata["features"][i],
                    dtype=DTYPE_REVERSE_MAPPING[
                        evset.dtypes[evset.feature_names[feat_pos]]
                    ],
                )
                for i, feat_pos in enumerate(dst_feat_pos)
            ],
            np.array(metadata["timestamps"], dtype=float),
        )
        for dst_key, metadata in dst_idx_metadata.items()
    }
    # finally, sort according to timestamps. TODO: this is merging sorted
    # arrays, we should later improve this code by avoiding the full sort
    for dst_idx_data in dst_evset.values():
        sorted_idxs = np.argsort(dst_idx_data.timestamps, kind="mergesort")
        # features
        for i, feature in enumerate(dst_idx_data.features):
            dst_idx_data.features[i] = feature[sorted_idxs]

        # timestamp
        dst_idx_data.timestamps = dst_idx_data.timestamps[sorted_idxs]

    return EventSet(
        dst_evset,
        feature_names=dst_feat_names,
        index_names=set_feat_names,
        is_unix_timestamp=evset.is_unix_timestamp,
    )


implementation_lib.register_operator_implementation(
    SetIndexOperator, SetIndexNumpyImplementation
)
