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

from temporian.core.operators.window.moving_count import (
    MovingCountOperator,
)
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.window.base import (
    BaseWindowNumpyImplementation,
)
from temporian.implementation.numpy_cc.operators import operators_cc


class MovingCountNumpyImplementation(BaseWindowNumpyImplementation):
    """Numpy implementation of the moving count operator."""

    def _implementation(self):
        return operators_cc.moving_count

    def _compute(
        self,
        src_timestamps: np.ndarray,
        src_features: List[np.ndarray],
        sampling_timestamps: np.ndarray,
        dst_features: List[np.ndarray],
    ) -> None:
        assert isinstance(self.operator, MovingCountOperator)

        del src_features  # Features are ignored

        kwargs = {
            "evset_timestamps": src_timestamps,
            "window_length": self.operator.window_length,
        }
        if self.operator.has_sampling:
            kwargs["sampling_timestamps"] = sampling_timestamps
        dst_feature = operators_cc.moving_count(**kwargs)
        dst_features.append(dst_feature)


implementation_lib.register_operator_implementation(
    MovingCountOperator, MovingCountNumpyImplementation
)
