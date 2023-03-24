"""Implementation for the simple moving average operator."""


# TODO: Implement in c++.
# TODO: Create a base class or utility tools to handle window operators.


from typing import Dict, Optional

import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.core.data.duration import Duration
from temporian.core.operators.window.simple_moving_average import (
    SimpleMovingAverageOperator,
)
from temporian.implementation.numpy import implementation_lib


class SimpleMovingAverageNumpyImplementation:
    """Numpy implementation for the simple moving average operator."""

    def __init__(self, op: SimpleMovingAverageOperator) -> None:
        assert isinstance(op, SimpleMovingAverageOperator)
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


implementation_lib.register_operator_implementation(
    SimpleMovingAverageOperator, SimpleMovingAverageNumpyImplementation
)


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

    # Replace masked values with NaN
    cross_product[
        mask == False  # pylint: disable=singleton-comparison
    ] = np.nan

    # Calculate the mean using numpy.nanmean
    mean = np.nanmean(cross_product, axis=1)

    return mean
