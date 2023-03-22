from collections import defaultdict
from typing import Dict, List, Union

import numpy as np

from temporian.core.operators.drop_index import DropIndexOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling

IndexMetadata = Dict[
    str, Union[int, List[List[NumpyFeature]], List[np.ndarray]]
]


def _impl(
    event: NumpyEvent, drop_idx_names: List[str], keep: bool
) -> NumpyEvent:
    # source index names
    src_idx_names = event.sampling.index

    # source sampling data
    src_samp_data = event.sampling.data

    # destination index names
    dst_idx_names = [
        name for name in event.sampling.index if name not in drop_idx_names
    ]
    # destination feature names
    dst_feat_names = (
        drop_idx_names + event.feature_names if keep else event.feature_names
    )
    # destination feature dtypes
    dst_dtypes = (
        {**event.sampling.dtypes, **event.dtypes} if keep else event.dtypes
    )
    # index positions to keep
    keep_idx_pos = [
        pos
        for pos, idx_name in enumerate(src_idx_names)
        if idx_name in dst_idx_names
    ]
    # index positions to drop
    drop_idx_pos = [
        pos
        for pos, idx_name in enumerate(src_idx_names)
        if idx_name not in dst_idx_names
    ]
    # intialize empty dict mapping destination index levels to block lengths,
    # features, and timestamps
    dst_idx_metadata: Dict[tuple, IndexMetadata] = defaultdict(
        lambda: {"block_length": 0, "timestamps": [], "features": []}
    )

    # loop over source index levels gathering destination index metadata
    for src_idx_lvl, timestamps in src_samp_data.items():
        # destination index level
        dst_idx_lvl = tuple((src_idx_lvl[i] for i in keep_idx_pos))

        # number of samples in this index level
        this_block_length = len(timestamps)

        # collapse index level
        drop_feats = (
            [
                NumpyFeature(
                    idx_name,
                    np.array(
                        [src_idx_lvl[i]] * this_block_length,
                        dtype=dst_dtypes[idx_name],
                    ),
                )
                for idx_name, i in zip(drop_idx_names, drop_idx_pos)
            ]
            if keep
            else []
        )
        # store metadata
        dst_idx_metadata[dst_idx_lvl]["block_length"] += this_block_length
        dst_idx_metadata[dst_idx_lvl]["timestamps"].append(timestamps)
        dst_idx_metadata[dst_idx_lvl]["features"].append(
            drop_feats + event.data[src_idx_lvl]
        )

    # allocate memory for destination sampling & event
    # sampling
    dst_samp_data = {
        idx_lvl: np.empty(metadata["block_length"])
        for idx_lvl, metadata in dst_idx_metadata.items()
    }
    # event
    dst_event_data = {
        idx_lvl: [
            NumpyFeature(
                name=dst_feat_name,
                data=np.empty(
                    metadata["block_length"], dtype=dst_dtypes[dst_feat_name]
                ),
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

    # finally, sort according to timestamps. TODO: this is merging sorted
    # arrays, we should later improve this code by avoiding the full sort
    for dst_idx_lvl in dst_event_data.keys():
        sorted_idxs = np.argsort(dst_samp_data[dst_idx_lvl], kind="mergesort")

        # sampling
        dst_samp_data[dst_idx_lvl] = dst_samp_data[dst_idx_lvl][sorted_idxs]

        # event
        for feature in dst_event_data[dst_idx_lvl]:
            feature.data = feature.data[sorted_idxs]

    # create & return output NumpyEvent w/ its NumpySampling
    return NumpyEvent(
        dst_event_data, NumpySampling(dst_idx_names, dst_samp_data)
    )


class DropIndexNumpyImplementation:
    def __init__(self, operator: DropIndexOperator) -> None:
        self.operator = operator

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        # get attributes
        drop_idx_names = self.operator.attributes()["labels"]
        keep = self.operator.attributes()["keep"]

        return {"event": _impl(event, drop_idx_names, keep)}
