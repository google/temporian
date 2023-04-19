from collections import defaultdict
from typing import Dict, List

import numpy as np

from temporian.core.operators.set_index import SetIndexOperator
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.feature import DTYPE_REVERSE_MAPPING
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.implementation.numpy import implementation_lib


class SetIndexNumpyImplementation(OperatorImplementation):
    """
    A class that represents the implementation of the SetIndexOperator for
    NumpyEvent objects.

    Attributes:
        operator (SetIndexOperator): The SetIndexOperator object.
    """

    def __init__(self, operator: SetIndexOperator) -> None:
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        """
        Execute the SetIndexOperator to the given NumpyEvent and return the
        updated NumpyEvent.

        Args:
            event:
                The input NumpyEvent object.

        Returns:
            A dictionary containing the resulting NumpyEvent object with the key
            "event".
        """
        # get attributes
        feature_names = self.operator.attributes["feature_names"]
        append = self.operator.attributes["append"]

        if append:
            return {"event": _append_impl(event, feature_names)}

        return {"event": _set_impl(event, feature_names)}


def _append_impl(event: NumpyEvent, append_feat_names: List[str]) -> NumpyEvent:
    """
    Append the specified feature names to the index of the given NumpyEvent.

    Args:
        event:
            The input NumpyEvent object.
        append_feat_names:
            A list of feature names to append to the index.

    Returns:
        The resulting NumpyEvent object with the updated index.
    """
    # destination index names
    dst_idx_names = event.index_names + append_feat_names

    # positions of features that are going to be part of the destination index
    dst_idx_pos = [
        event.feature_names.index(append_feat_name)
        for append_feat_name in append_feat_names
    ]
    # positions of features that are being kept
    dst_feat_pos = {
        feat_name: pos
        for pos, feat_name in enumerate(event.feature_names)
        if feat_name not in append_feat_names
    }
    # initialize destination event & sampling data
    dst_event_data = {}
    for src_idx_lvl, src_idx_lvl_data in event.iterindex():
        dst_idx_suffs = [
            tuple(x)
            for x in zip(*[src_idx_lvl_data.features[i] for i in dst_idx_pos])
        ]
        for target_dst_idx_suff in set(dst_idx_suffs):
            # find all occurrences of destination suff in source event
            dst_idx_suff_pos = [
                idx
                for idx, dst_idx_suff in enumerate(dst_idx_suffs)
                if dst_idx_suff == target_dst_idx_suff
            ]
            # create destination index
            dst_idx_lvl = src_idx_lvl + tuple(target_dst_idx_suff)

            dst_event_data[dst_idx_lvl] = IndexData(
                [
                    event.data[src_idx_lvl].features[feat_pos][dst_idx_suff_pos]
                    for feat_name, feat_pos in dst_feat_pos.items()
                ],
                # fill sampling data
                event.data[src_idx_lvl].timestamps[dst_idx_suff_pos],
            )

    # finally, sort according to timestamps. TODO: this is merging sorted
    # arrays, we should later improve this code by avoiding the full sort
    for dst_index_data in dst_event_data.values():
        sorted_idxs = np.argsort(dst_index_data.timestamps, kind="mergesort")
        # features
        for i, feature in enumerate(dst_index_data.features):
            dst_index_data.features[i] = feature[sorted_idxs]

        # timestamp
        dst_index_data.timestamps = dst_index_data.timestamps[sorted_idxs]

    return NumpyEvent(
        dst_event_data,
        feature_names=list(dst_feat_pos),
        index_names=dst_idx_names,
        is_unix_timestamp=event.is_unix_timestamp,
    )


def _set_impl(event: NumpyEvent, set_feat_names: List[str]) -> NumpyEvent:
    """
    Set the specified feature names as the new index of the given NumpyEvent.

    Args:
        event:
            The input NumpyEvent object.
        set_feat_names:
            A list of feature names to set as the new index.

    Returns:
        The resulting NumpyEvent object with the updated index.
    """
    # positions of features that are going to be part of the destination index
    dst_idx_pos = [
        event.feature_names.index(set_feat_name)
        for set_feat_name in set_feat_names
    ]
    # positions of features that are being kept
    dst_feat_pos = {
        feat_name: pos
        for pos, feat_name in enumerate(event.feature_names)
        if feat_name not in set_feat_names
    }
    # intialize empty dict mapping destination index levels to block lengths
    dst_idx_metadata = defaultdict(
        lambda: {
            "timestamps": [],
            "features": {feat_name: [] for feat_name in dst_feat_pos},
        }
    )
    # loop over source index levels gathering destination indexes
    for src_idx_lvl, src_idx_lvl_data in event.iterindex():
        dst_idx_lvls = [
            tuple(x)
            for x in zip(*[src_idx_lvl_data.features[i] for i in dst_idx_pos])
        ]
        for i, dst_idx_lvl in enumerate(dst_idx_lvls):
            dst_idx_metadata[dst_idx_lvl]["timestamps"].append(
                event.data[src_idx_lvl].timestamps[i]
            )
            for feat_name, feat_pos in dst_feat_pos.items():
                dst_idx_metadata[dst_idx_lvl]["features"][feat_name].append(
                    src_idx_lvl_data.features[feat_pos][i]
                )

    # create destination event
    dst_event_data = {
        dst_idx_lvl: IndexData(
            [
                np.array(
                    metadata["features"][feat_name],
                    dtype=DTYPE_REVERSE_MAPPING[event.dtypes[feat_name]],
                )
                for feat_name in dst_feat_pos
            ],
            np.array(metadata["timestamps"], dtype=float),
        )
        for dst_idx_lvl, metadata in dst_idx_metadata.items()
    }
    # finally, sort according to timestamps. TODO: this is merging sorted
    # arrays, we should later improve this code by avoiding the full sort
    for dst_index_data in dst_event_data.values():
        sorted_idxs = np.argsort(dst_index_data.timestamps, kind="mergesort")
        # features
        for i, feature in enumerate(dst_index_data.features):
            dst_index_data.features[i] = feature[sorted_idxs]

        # timestamp
        dst_index_data.timestamps = dst_index_data.timestamps[sorted_idxs]

    return NumpyEvent(
        dst_event_data,
        feature_names=list(dst_feat_pos),
        index_names=set_feat_names,
        is_unix_timestamp=event.is_unix_timestamp,
    )


implementation_lib.register_operator_implementation(
    SetIndexOperator, SetIndexNumpyImplementation
)
