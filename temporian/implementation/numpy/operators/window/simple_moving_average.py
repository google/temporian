"""Implementation for the SimpleMovingAverage operator."""

from typing import Dict, Optional

import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.core.data.duration import Duration
from temporian.core.operators.simple_moving_average import SimpleMovingAverage


class SimpleMovingAverageNumpyImplementation:
    """Numpy implementation for the simple moving average operator."""

    def __init__(self, op: SimpleMovingAverage) -> None:
        assert isinstance(op, SimpleMovingAverage)
        self._op = op

    def __call__(
        self,
        event: NumpyEvent,
        sampling: Optional[NumpyEvent] = None,
    ) -> Dict[str, NumpyEvent]:
        # TODO: Add support for sampling.
        if sampling is not None:
            raise NotImplementedError("Sampling not implemented")

        dst_event = NumpyEvent(data={}, sampling=event.sampling)

        # For each index
        # Note: mts => multi time series
        # Note: ts => time series
        for index, src_mts in event.data.items():
            dst_mts = []
            dst_event.data[index] = dst_mts
            src_timestamps = event.sampling.data[index]

            # For each feature
            for src_ts in src_mts:
                dst_feature_name = f"sma_{src_ts.name}"
                dst_ts_data = _impl(
                    src_ts.data, src_timestamps, self._op.window_length()
                )
                dst_mts.append(NumpyFeature(dst_feature_name, dst_ts_data))

        return {"event": dst_event}


def _impl(src: np.array, timestamps: np.array, win: Duration) -> np.array:
    # TODO: Implement in c++.
    # TODO: Add support for missing (NaN) input values.
    # TODO: Both sides are currently inclusive. Check if this makes sense.

    # This implementation is simple but expensive. It will create multiple
    # O(n^2) arrays, where n is the number of time samples.

    # Right side
    r = timestamps[:, np.newaxis] >= timestamps[np.newaxis, :]
    # Left side
    l = timestamps[:, np.newaxis] <= (timestamps[np.newaxis, :] + win)
    m = r & l
    dst = np.sum(src * m, axis=1) / np.sum(m, axis=1)

    return dst
