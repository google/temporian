from typing import Dict, List

import numpy as np

from temporian.implementation.numpy.data.event import NumpyFeature


def _sort_by_timestamp(
    event_data: Dict[str, List[NumpyFeature]], samp_data: Dict[str, np.array]
) -> None:
    for idx_lvl in event_data:
        sorted_idxs = np.argsort(samp_data[idx_lvl], kind="mergesort")

        # sampling
        samp_data[idx_lvl] = samp_data[idx_lvl][sorted_idxs]

        # event
        for feature in event_data[idx_lvl]:
            feature.data = feature.data[sorted_idxs]
