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


class DropIndexNumpyImplementation:
    def __init__(self, operator: DropIndexOperator) -> None:
        """
        Initializes an instance of the DropIndexNumpyImplementation class.

        Args:
            operator: The DropIndexOperator instance.
        """
        self.operator = operator

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        """
        Calls the implementation function with the specified event.

        Args:
            event: The input NumpyEvent.

        Returns:
            A dictionary with the modified NumpyEvent.
        """
        # get attributes from operator
        drop_index_names = self.operator.index_names
        keep = self.operator.keep
        dst_index_names = self.operator.dst_index_names
        dst_feat_names = self.operator.dst_feat_names

        # source index names
        src_index_names = event.sampling.index

        # destination feature dtypes
        dst_dtypes = (
            {**event.sampling.dtypes, **event.dtypes} if keep else event.dtypes
        )
        # index positions to keep
        keep_index_pos = [
            pos
            for pos, idx_name in enumerate(src_index_names)
            if idx_name in dst_index_names
        ]
        # index positions to drop
        drop_index_pos = [
            pos
            for pos, idx_name in enumerate(src_index_names)
            if idx_name not in dst_index_names
        ]
        # intialize empty dict mapping destination index levels to block lengths,
        # features, and timestamps
        dst_index_metadata: Dict[tuple, IndexMetadata] = defaultdict(
            lambda: {"block_length": 0, "timestamps": [], "features": []}
        )
        # loop over source index levels gathering destination index metadata
        for src_index_lvl, timestamps in event.sampling.data.items():
            # destination index level
            dst_index_lvl = tuple((src_index_lvl[i] for i in keep_index_pos))

            # number of samples in this index level
            this_block_length = len(timestamps)

            # collapse index level
            drop_feats = (
                [
                    NumpyFeature(
                        idx_name,
                        np.array(
                            [src_index_lvl[i]] * this_block_length,
                            dtype=dst_dtypes[idx_name],
                        ),
                    )
                    for idx_name, i in zip(drop_index_names, drop_index_pos)
                ]
                if keep
                else []
            )
            # store metadata
            dst_index_metadata[dst_index_lvl][
                "block_length"
            ] += this_block_length
            dst_index_metadata[dst_index_lvl]["timestamps"].append(timestamps)
            dst_index_metadata[dst_index_lvl]["features"].append(
                drop_feats + event.data[src_index_lvl]
            )

        # allocate memory for destination sampling & event
        # sampling
        dst_samp_data = {
            index_lvl: np.empty(metadata["block_length"], dtype=np.float64)
            for index_lvl, metadata in dst_index_metadata.items()
        }
        # event
        dst_event_data = {
            index_lvl: [
                NumpyFeature(
                    name=dst_feat_name,
                    data=np.empty(
                        metadata["block_length"],
                        dtype=dst_dtypes[dst_feat_name],
                    ),
                )
                for dst_feat_name in dst_feat_names
            ]
            for index_lvl, metadata in dst_index_metadata.items()
        }
        # assign data to previously allocated memory
        for dst_index_lvl, metadata in dst_index_metadata.items():
            ptr = 0
            for timestamps, feats in zip(
                metadata["timestamps"], metadata["features"]
            ):
                # number of samples in this index level
                this_block_length = len(timestamps)

                # allocate sampling
                dst_samp_data[dst_index_lvl][
                    ptr : ptr + this_block_length
                ] = timestamps

                # allocate event
                for i, feat in enumerate(feats):
                    dst_event_data[dst_index_lvl][i].data[
                        ptr : ptr + this_block_length
                    ] = feat.data

                # increment pointer
                ptr += this_block_length

        # finally, sort according to timestamps. TODO: this is merging sorted
        # arrays, we should later improve this code by avoiding the full sort
        for dst_index_lvl in dst_event_data.keys():
            sorted_idxs = np.argsort(
                dst_samp_data[dst_index_lvl], kind="mergesort"
            )
            # sampling
            dst_samp_data[dst_index_lvl] = dst_samp_data[dst_index_lvl][
                sorted_idxs
            ]

            # event
            for feature in dst_event_data[dst_index_lvl]:
                feature.data = feature.data[sorted_idxs]

        # create & return output NumpyEvent w/ its NumpySampling
        return {
            "event": NumpyEvent(
                dst_event_data, NumpySampling(dst_index_names, dst_samp_data)
            )
        }
