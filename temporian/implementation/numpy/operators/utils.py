from typing import Dict, List

import numpy as np

from temporian.implementation.numpy.data.event import NumpyFeature


def _sort_by_timestamp(
    event_data: Dict[str, List[NumpyFeature]], samp_data: Dict[str, np.array]
) -> None:
    """Sorts the data in event_data and samp_data according to their
    timestamps. This operations is done inplace.

    This function takes in two dictionaries, event and samp_data,
    that contain event data and sampling data, respectively. It sorts the data
    in both dictionaries based on the timestamps in samp_data using a
    "mergesort" algorithm.

    Args:
        event_data:
            A dictionary containing the event data, with keys representing index
            levels and values being lists of NumpyFeature objects.
        samp_data:
            A dictionary containing the sampling data, with keys representing
            index levels and values being numpy arrays of timestamps.
    """
    for idx_lvl in event_data:
        sorted_idxs = np.argsort(samp_data[idx_lvl], kind="mergesort")

        # sampling
        samp_data[idx_lvl] = samp_data[idx_lvl][sorted_idxs]

        # event
        for feature in event_data[idx_lvl]:
            feature.data = feature.data[sorted_idxs]
