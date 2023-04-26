# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Dict, Optional

import numpy as np
from temporian.core.data.duration import Duration
from temporian.core.operators.window.base import BaseWindowOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseWindowNumpyImplementation(OperatorImplementation):
    """Interface definition and common logic for numpy implementation of
    window operators."""

    def __init__(self, operator: BaseWindowOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, BaseWindowOperator)

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

            mask = self._build_accumulator_mask(
                src_timestamps,
                sampling_timestamps,
                self.operator.window_length(),
            )

            # For each feature
            for src_ts in src_mts:
                dst_feature_name = f"{self.operator.prefix}_{src_ts.name}"
                dst_ts_data = self._apply_accumulator_mask(src_ts.data, mask)
                dst_mts.append(NumpyFeature(dst_feature_name, dst_ts_data))

        return {"event": dst_event}

    def _build_accumulator_mask(
        self,
        data_timestamps: np.array,
        sampling_timestamps: np.array,
        win: Duration,
    ) -> np.array:
        """Creates a 2d boolean matrix containing the summing instructions.

        Returns:
            Matrix `m`, where `m[i,j]` is true iif input value `j` is averaged
            in the output value `i`.
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

    def _apply_accumulator_mask(
        self, src: np.array, mask: np.array
    ) -> np.array:
        """Sums elements according to an accumulator mask."""

        # Broadcast of feature values to requested timestamps.
        cross_product = src * mask

        # Replace masked values with NaN
        cross_product[
            mask == False
        ] = np.nan  # pylint: disable=singleton-comparison

        return self._apply_operation(cross_product)

    @abstractmethod
    def _apply_operation(self, values: np.array) -> np.array:
        """Reduces each of the rows in an array of values to a single value.

        The input array has a shape (n, m), where n is the length of the feature
        and m is the size of the window. Each row represents a window of data
        points, with NaN values used for padding when the window size is smaller
        than the number of data points in the time series.

        Args:
            values: Input array, of shape (n, m).

        Returns:
            1D NumPy array of shape (n,) containing the result for each row in
            the input array.
        """
