from collections import defaultdict
from typing import Dict, List

import numpy as np

from temporian.core.operators.set_index import SetIndexOperator
from temporian.implementation.numpy.data.feature import DTYPE_REVERSE_MAPPING
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy.operators.utils import _sort_by_timestamp
from temporian.implementation.numpy.operators.base import OperatorImplementation


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
    dst_idx_names = event.sampling.index + append_feat_names

    # positions of features that are going to be part of the destination index
    dst_idx_pos = [
        event.feature_names().index(append_feat_name)
        for append_feat_name in append_feat_names
    ]
    # positions of features that are being kept
    dst_feat_pos = {
        feat_name: pos
        for pos, feat_name in enumerate(event.feature_names())
        if feat_name not in append_feat_names
    }
    # initialize destination event & sampling data
    dst_event_data = {}
    dst_samp_data = {}
    for src_idx_lvl, src_idx_lvl_feats in event.data.items():
        dst_idx_suffs = [
            tuple(x)
            for x in zip(*[src_idx_lvl_feats[i].data for i in dst_idx_pos])
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

            dst_event_data[dst_idx_lvl] = [
                NumpyFeature(
                    feat_name,
                    event.data[src_idx_lvl][feat_pos].data[dst_idx_suff_pos],
                )
                for feat_name, feat_pos in dst_feat_pos.items()
            ]
            # fill sampling data
            dst_samp_data[dst_idx_lvl] = event.sampling.data[src_idx_lvl][
                dst_idx_suff_pos
            ]

    # finally, sort according to timestamps. TODO: this is merging sorted
    # arrays, we should later improve this code by avoiding the full sort
    _sort_by_timestamp(dst_event_data, dst_samp_data)

    return NumpyEvent(
        dst_event_data, NumpySampling(dst_idx_names, dst_samp_data)
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
        event.feature_names().index(append_feat_name)
        for append_feat_name in set_feat_names
    ]
    # positions of features that are being kept
    dst_feat_pos = {
        feat_name: pos
        for pos, feat_name in enumerate(event.feature_names())
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
    for src_idx_lvl, src_idx_lvl_feats in event.data.items():
        dst_idx_lvls = [
            tuple(x)
            for x in zip(*[src_idx_lvl_feats[i].data for i in dst_idx_pos])
        ]
        for i, dst_idx_lvl in enumerate(dst_idx_lvls):
            dst_idx_metadata[dst_idx_lvl]["timestamps"].append(
                event.sampling.data[src_idx_lvl][i]
            )
            for feat_name, feat_pos in dst_feat_pos.items():
                dst_idx_metadata[dst_idx_lvl]["features"][feat_name].append(
                    src_idx_lvl_feats[feat_pos].data[i]
                )

    # create destination sampling & event
    # sampling
    dst_samp_data = {
        dst_idx_lvl: np.array(metadata["timestamps"], dtype=float)
        for dst_idx_lvl, metadata in dst_idx_metadata.items()
    }
    # event
    dst_event_data = {
        dst_idx_lvl: [
            NumpyFeature(
                name=feat_name,
                data=np.array(
                    metadata["features"][feat_name],
                    dtype=DTYPE_REVERSE_MAPPING[event.dtypes[feat_name]],
                ),
            )
            for feat_name in dst_feat_pos
        ]
        for dst_idx_lvl, metadata in dst_idx_metadata.items()
    }
    # finally, sort according to timestamps. TODO: this is merging sorted
    # arrays, we should later improve this code by avoiding the full sort
    _sort_by_timestamp(dst_event_data, dst_samp_data)

    return NumpyEvent(
        dst_event_data, NumpySampling(set_feat_names, dst_samp_data)
    )
