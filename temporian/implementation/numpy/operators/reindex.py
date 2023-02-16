from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from temporian.core.operators.reindex import ReIndex
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


def _subset_impl(
    src_event: NumpyEvent,
    src_idx_names: List[str],
    dst_idx_names: List[str],
):
    # source sampling data
    src_samp_data = src_event.sampling.data

    # index positions to keep
    keep_idx_pos = [
        pos
        for pos, index_name in enumerate(src_idx_names)
        if index_name in dst_idx_names
    ]
    # index positions to drop
    drop_idx_pos = [
        pos
        for pos, index_name in enumerate(src_idx_names)
        if index_name not in dst_idx_names
    ]
    # dropped from index feature names
    drop_feat_names = [src_idx_names[i] for i in drop_idx_pos]

    # intialize empty dict mapping destination index levels to block lengths,
    # features, and timestamps
    dst_idx_metadata: Dict[
        tuple, Dict[str, int | List[NumpyFeature] | np.array]
    ] = {}

    # loop over source index levels gathering destination index metadata
    for src_idx_lvl, timestamps in src_samp_data.items():
        # destination index level
        dst_idx_lvl = tuple((src_idx_lvl[i] for i in keep_idx_pos))

        # use try - except here. Could instead check if key already exists in dict,
        # but it's likely much slower
        try:
            # number of samples in this index level
            this_block_length = len(timestamps)

            # collapse index level
            drop_feats = [
                NumpyFeature(
                    feat_name, np.array([src_idx_lvl[i]] * this_block_length)
                )
                for feat_name, i in zip(drop_feat_names, drop_idx_pos)
            ]
            # store metadata
            dst_idx_metadata[dst_idx_lvl]["block_length"] += this_block_length
            dst_idx_metadata[dst_idx_lvl]["timestamps"].append(timestamps)
            dst_idx_metadata[dst_idx_lvl]["features"].append(
                drop_feats + src_event.data[src_idx_lvl]
            )

        except KeyError:
            # first time destination index level is encountered - create entry
            dst_idx_metadata[dst_idx_lvl] = {}
            dst_idx_metadata[dst_idx_lvl]["block_length"] = this_block_length
            dst_idx_metadata[dst_idx_lvl]["timestamps"] = [timestamps]
            dst_idx_metadata[dst_idx_lvl]["features"] = [
                drop_feats + src_event.data[src_idx_lvl]
            ]

    # allocate memory for destination sampling & event
    # sampling
    dst_samp_data = {
        idx_lvl: np.empty(metadata["block_length"])
        for idx_lvl, metadata in dst_idx_metadata.items()
    }
    # destination feature names
    dst_feat_names = drop_feat_names + src_event.feature_names

    # event
    dst_event_data = {
        idx_lvl: [
            NumpyFeature(
                name=dst_feat_name, data=np.empty(metadata["block_length"])
            )
            for dst_feat_name in dst_feat_names
        ]
        for idx_lvl, metadata in dst_idx_metadata.items()
    }
    # assign data to previously allocated memory
    for dst_idx_lvl, metadata in dst_idx_metadata.items():
        ptr = 0
        for timestamps, feats in zip(
            metadata["timestamps"], metadata["features"]
        ):
            # number of samples in this index level
            this_block_length = len(timestamps)

            # allocate sampling
            dst_samp_data[dst_idx_lvl][
                ptr : ptr + this_block_length
            ] = timestamps

            # allocate event
            for i, feat in enumerate(feats):
                dst_event_data[dst_idx_lvl][i].data[
                    ptr : ptr + this_block_length
                ] = feat.data

            # increment pointer
            ptr += this_block_length

    # finally, sort according to timestamps
    for dst_idx_lvl in dst_event_data.keys():
        sorted_idxs = np.argsort(dst_samp_data[dst_idx_lvl], kind="mergesort")

        # sampling
        dst_samp_data[dst_idx_lvl] = dst_samp_data[dst_idx_lvl][sorted_idxs]

        # event
        for feature in dst_event_data[dst_idx_lvl]:
            feature.data = feature.data[sorted_idxs]

    # create & return output NumpyEvent w/ it's NumpySampling
    return NumpyEvent(
        dst_event_data, NumpySampling(src_idx_names, dst_samp_data)
    )


class ReIndexNumpyImplementation:
    """Select a subset of features from an event."""

    def __init__(self, op: ReIndex) -> None:
        assert isinstance(op, ReIndex)
        self._op = op

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        # sampling index names
        src_index_names = event.sampling.names
        dst_index_names = self._op.attributes()["dst_index"]

        # fist case - destination index is a subset of source index
        if set(dst_index_names).issubset(set(src_index_names)):
            output_event = _subset_impl(event, src_index_names, dst_index_names)

        elif set(dst_index_names).issuperset(set(src_index_names)):
            pass

        return {"event": output_event}
