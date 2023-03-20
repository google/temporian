from typing import Dict

import numpy as np

from temporian.core.operators.reset_index import ResetIndexOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


class ResetIndexNumpyImplementation:
    def __init__(self, operator: ResetIndexOperator) -> None:
        self.operator = operator

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        # source index names
        src_idx_names = event.sampling.index

        # length of source index
        src_idx_len = len(src_idx_names)

        # number of destination timestamps
        num_dst_timestamps = sum(
            [
                len(src_timestamps)
                for src_timestamps in event.sampling.data.values()
            ]
        )
        # allocate memory for destination features & sampling
        dst_dtypes = {**event.sampling.dtypes, **event.dtypes}
        dst_features = [
            NumpyFeature(
                feat_name,
                np.empty(num_dst_timestamps, dtype=dst_dtypes[feat_name]),
            )
            for feat_name in src_idx_names + event.feature_names
        ]
        dst_sampling_data = np.zeros(num_dst_timestamps)

        # iterate over source event
        ptr = 0
        for src_idx_lvl, src_timestamps in event.sampling.data.items():
            num_timestamps = len(src_timestamps)

            # sampling
            dst_sampling_data[ptr : ptr + num_timestamps] = src_timestamps

            # features:
            # 1. from source sampling (expanded)
            for i, src_idx_lvl_elem in enumerate(src_idx_lvl):
                dst_features[i].data[ptr : ptr + num_timestamps] = [
                    src_idx_lvl_elem
                ] * num_timestamps

            # 2. from already existing features (appended)
            for i, src_feat in enumerate(event.data[src_idx_lvl]):
                dst_features[i + src_idx_len].data[
                    ptr : ptr + num_timestamps
                ] = src_feat.data

            ptr += num_timestamps

        # finally, sort data according to timestamps (increasingly)
        sorted_idxs = np.argsort(dst_sampling_data, kind="mergesort")

        # sampling
        dst_sampling_data = dst_sampling_data[sorted_idxs]

        # features
        for dst_feat in dst_features:
            dst_feat.data = dst_feat.data[sorted_idxs]

        # create & return output NumpyEvent w/ its NumpySampling
        return {
            "event": NumpyEvent(
                {(): dst_features},
                NumpySampling(index=[], data={(): dst_sampling_data}),
            )
        }
