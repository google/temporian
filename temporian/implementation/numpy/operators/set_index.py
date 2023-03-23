from collections import defaultdict
from typing import Dict, List, Union

import numpy as np

from temporian.core.operators.set_index import SetIndexOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling

IndexMetadata = Dict[
    str, Union[int, List[List[NumpyFeature]], List[np.ndarray]]
]


def _append_impl(event: NumpyEvent, append_feat_names: List[str]) -> NumpyEvent:
    # destination index names
    dst_idx_names = event.sampling.index + append_feat_names

    # positions of features that are going to be part of the destination index
    dst_idx_pos = [
        event.feature_names.index(append_feat_name)
        for append_feat_name in append_feat_names
    ]
    # positions of features that are being kept
    keep_feats_pos = [
        pos
        for pos, feat_name in enumerate(event.feature_names)
        if feat_name not in append_feat_names
    ]
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

            # initialize event & sampling destination entries
            dst_event_data[dst_idx_lvl] = []
            # fill event data
            for idx in keep_feats_pos:
                dst_event_data[dst_idx_lvl].append(
                    NumpyFeature(
                        event.feature_names[idx],
                        event.data[src_idx_lvl][idx].data[dst_idx_suff_pos],
                    )
                )
            # fill sampling data
            dst_samp_data[dst_idx_lvl] = event.sampling.data[src_idx_lvl][
                dst_idx_suff_pos
            ]
            # finally, sort according to timestamps. TODO: this is merging sorted
            # arrays, we should later improve this code by avoiding the full sort
            for dst_idx_lvl in dst_event_data.keys():
                sorted_idxs = np.argsort(
                    dst_samp_data[dst_idx_lvl], kind="mergesort"
                )
                # sampling
                dst_samp_data[dst_idx_lvl] = dst_samp_data[dst_idx_lvl][
                    sorted_idxs
                ]
                # event
                for feature in dst_event_data[dst_idx_lvl]:
                    feature.data = feature.data[sorted_idxs]

    return NumpyEvent(
        dst_event_data, NumpySampling(dst_idx_names, dst_samp_data)
    )


class SetIndexNumpyImplementation:
    def __init__(self, operator: SetIndexOperator) -> None:
        self.operator = operator

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        # get attributes
        labels = self.operator.attributes()["labels"]
        append = self.operator.attributes()["append"]

        if append:
            return {"event": _append_impl(event, labels)}
