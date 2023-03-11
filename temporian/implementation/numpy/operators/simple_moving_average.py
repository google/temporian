"""Implementation for the SimpleMovingAverage operator."""


# TODO: Implement in c++.
# TODO: Add support for missing (NaN) input values.


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
        if sampling is None:
            sampling = event

        dst_event = NumpyEvent(data={}, sampling=sampling.sampling)

        # Naming convention:
        #   mts => multi time series
        #   ts => time series
        #   dst => destination
        #   src => source

        # For each index
        for index, src_mts in event.data.items():
            dst_mts = []
            dst_event.data[index] = dst_mts
            src_timestamps = event.sampling.data[index]
            sampling_timestamps = sampling.sampling.data[index]

            mask = _build_accumulator_mask(
                src_timestamps, sampling_timestamps, self._op.window_length()
            )

            # For each feature
            for src_ts in src_mts:
                dst_feature_name = f"sma_{src_ts.name}"
                dst_ts_data = _apply_accumulator_mask(src_ts.data, mask)
                dst_mts.append(NumpyFeature(dst_feature_name, dst_ts_data))

        return {"event": dst_event}


def _build_accumulator_mask(
    data_timestamps: np.array, sampling_timestamps: np.array, win: Duration
) -> np.array:
    """Creates a 2d boolean matrix containing the summing instructions.

    The returned matrix "m[i,j]" is defined as:
        m[i,j] is true iif. input value "j" is averaged in the output value "i".
    """

    # This implementation is simple but expensive. It will create multiple
    # O(n^2) arrays, where n is the number of time samples.

    right_side = (
        sampling_timestamps[:, np.newaxis] >= data_timestamps[np.newaxis, :]
    )

    # TODO: Make left side inclusivity/exclusivity a parameter.
    left_side = sampling_timestamps[:, np.newaxis] <= (
        data_timestamps[np.newaxis, :] + win
    )

    return right_side & left_side


def _apply_accumulator_mask(src: np.array, mask: np.array) -> np.array:
    """Sums elements according to an accumulator mask."""

    # Broadcast of feature values to requested timestamps.
    cross_product = src * mask

    # Ignore nan (i.e. missing) values
    nan_mask = np.isnan(cross_product)
    cross_product[nan_mask] = 0.0
    mask[nan_mask] = False

    sum_values = np.sum(cross_product, axis=1)
    count_values = np.sum(mask, axis=1)

    # TODO: Find a better way to hide divisions by zero without warnings.
    mean = sum_values / np.maximum(1, count_values)
    mean[np.equal(count_values, 0)] = np.nan

    return mean
