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
import numpy as np


from typing import List, Union, Optional, Dict
from temporian.core.data.duration_utils import NormalizedDuration
from temporian.core.operators.window.moving_quantile import (
    MovingQuantileOperator,
)
from temporian.implementation.numpy.data.event_set import (
    EventSet,
)
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.window.base import (
    BaseWindowNumpyImplementation,
)
from temporian.implementation.numpy_cc.operators import operators_cc


class MovingQuantileNumpyImplementation(BaseWindowNumpyImplementation):
    """Numpy implementation of the simple moving average operator."""

    def _implementation(self):
        return operators_cc.moving_quantile

    def _compute(
        self,
        src_timestamps: np.ndarray,
        src_features: List[np.ndarray],
        sampling_timestamps: Optional[np.ndarray],
        dst_features: List[np.ndarray],
        window_length: Union[NormalizedDuration, np.ndarray],
    ) -> None:
        assert isinstance(self.operator, MovingQuantileOperator)

        implementation = self._implementation()
        for src_ts in src_features:
            kwargs = {
                "evset_timestamps": src_timestamps,
                "evset_values": src_ts,
                "window_length": window_length,
                "quantile": self.operator.quantile,
            }
            if sampling_timestamps is not None:
                kwargs["sampling_timestamps"] = sampling_timestamps
            dst_feature = implementation(**kwargs)
            dst_features.append(dst_feature)


implementation_lib.register_operator_implementation(
    MovingQuantileOperator, MovingQuantileNumpyImplementation
)
