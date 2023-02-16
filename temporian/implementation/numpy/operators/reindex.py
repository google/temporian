from copy import deepcopy
from typing import Dict, List

import numpy as np

from temporian.core.operators.reindex import ReIndex
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


class ReIndexNumpyImplementation:
    """Select a subset of features from an event."""

    def __init__(self, op: ReIndex) -> None:
        assert isinstance(op, ReIndex)
        self._op = op

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        # sampling index names
        src_index = event.sampling.names
        dst_index = self._op.attributes()["dst_index"]

        # index positions to keep
        dst_index_positions = [
            pos
            for pos, index_name in enumerate(event.sampling.names)
            if index_name in dst_index
        ]
        # index positions to collapse
        src_index_positions = [
            pos
            for pos, index_name in enumerate(event.sampling.names)
            if index_name not in dst_index
        ]
        # fist case - destination index is a subset of source index
        if set(dst_index).issubset(set(src_index)):
            # initialize data dictionaries
            dst_sampling_data = {}
            dst_event_data = {}

            # loop over source index levels
            for src_index_level in event.data.keys():
                # kept index names
                dst_index_level = tuple(
                    (src_index_level[i] for i in dst_index_positions)
                )
                # sampling
                # get source sampling
                timestamps = event.sampling.data[src_index_level]
                try:
                    dst_sampling_data[dst_index_level] = np.append(
                        dst_sampling_data[dst_index_level], timestamps
                    )
                except KeyError:
                    # create index level
                    dst_sampling_data[dst_index_level] = np.array(timestamps)

                # event
                # get source features
                features = event.data[src_index_level]

                # index names from source index that are going to be converted
                # to features
                dst_features = [
                    feature_name
                    for feature_name in src_index
                    if feature_name not in dst_index
                ]
                try:
                    # append collapsed index levels first
                    for i, j in enumerate(src_index_positions):
                        dst_event_data[dst_index_level][i].data = np.append(
                            dst_event_data[dst_index_level][i].data,
                            [src_index_level[j]] * len(features[0].data),
                        )
                    # append other feature data
                    for i, feature in enumerate(
                        features, start=len(dst_features)
                    ):
                        dst_event_data[dst_index_level][i].data = np.append(
                            dst_event_data[dst_index_level][i].data,
                            feature.data,
                        )
                except KeyError:
                    # create index level
                    dst_event_data[dst_index_level] = [
                        NumpyFeature(
                            feature_name,
                            None,  # sampling=None, removed in pending PRs
                            np.array(
                                [src_index_level[i]] * len(features[0].data)
                            ),
                        )
                        for i, feature_name in zip(
                            src_index_positions, dst_features
                        )
                    ] + deepcopy(features)

            # sort data and sampling according to timestamps
            for dst_index_level in dst_event_data.keys():
                sorted_idxs = np.argsort(dst_sampling_data[dst_index_level])

                # sampling
                dst_sampling_data[dst_index_level] = dst_sampling_data[
                    dst_index_level
                ][sorted_idxs]

                # event
                for feature in dst_event_data[dst_index_level]:
                    feature.data = feature.data[sorted_idxs]

            output_event = NumpyEvent(
                data=dst_event_data,
                sampling=NumpySampling(dst_index, dst_sampling_data),
            )

        elif set(dst_index).issuperset(set(src_index)):
            pass

        return {"event": output_event}
