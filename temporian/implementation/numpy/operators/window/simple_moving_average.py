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

from typing import List

import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy import implementation_lib
from temporian.core.operators.window.simple_moving_average import (
    SimpleMovingAverageOperator,
)
from temporian.implementation.numpy.operators.window.base import (
    BaseWindowNumpyImplementation,
)
from temporian.implementation.numpy_cc.operators import window as window_cc
from temporian.core.data import dtype

FN_MAP = {
    dtype.FLOAT32: window_cc.simple_moving_average_float32,
    dtype.FLOAT64: window_cc.simple_moving_average_float64,
}


class SimpleMovingAverageNumpyImplementation(BaseWindowNumpyImplementation):
    """Numpy implementation of the simple moving average operator."""

    def __init__(self, operator: SimpleMovingAverageOperator) -> None:
        super().__init__(operator)

    def _compute(
        self,
        src_timestamps: np.ndarray,
        src_features: List[NumpyFeature],
        sampling_timestamps: np.ndarray,
        dst_features: List[NumpyFeature],
    ):
        for src_ts in src_features:
            fn = window_cc.simple_moving_average_float64
            args = {
                "event_timestamps": src_timestamps,
                "event_values": src_ts.data,
                "window_length": self.operator.window_length(),
            }
            if self.operator.has_sampling():
                args["sampling_timestamps"] = sampling_timestamps
            dst_feature = fn(**args)
            print("@@@@@@@@ dst_feature:", dst_feature, flush=True)
            dst_features.append(NumpyFeature(src_ts.name, dst_feature))

    def _apply_operation(self, values: np.array) -> np.array:
        raise NotImplementedError()


implementation_lib.register_operator_implementation(
    SimpleMovingAverageOperator, SimpleMovingAverageNumpyImplementation
)
