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

from typing import List, Optional, Union
import numpy as np
from temporian.core.data.duration_utils import NormalizedDuration

from temporian.core.operators.window.moving_count import (
    MovingCountOperator,
)
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.data.dtype_normalization import (
    tp_dtype_to_np_dtype,
)
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
        sampling_timestamps: Optional[np.ndarray],
        dst_features: List[np.ndarray],
        window_length: Union[NormalizedDuration, np.ndarray],
    ) -> None:
        assert isinstance(self.operator, MovingCountOperator)

        del src_features  # Features are ignored

        implementation = self._implementation()

        kwargs = {
            "evset_timestamps": src_timestamps,
            "window_length": window_length,
        }
        if sampling_timestamps is not None:
            kwargs["sampling_timestamps"] = sampling_timestamps
        dst_feature = implementation(**kwargs)
        dst_features.append(dst_feature)

    def apply_feature_wise(
        self,
        src_timestamps: np.ndarray,
        src_feature: np.ndarray,
        feature_idx: int,
    ) -> np.ndarray:
        """Applies the operator on a single feature."""

        assert isinstance(self.operator, MovingCountOperator)

        implementation = self._implementation()
        kwargs = {
            "evset_timestamps": src_timestamps,
            "window_length": self.operator.window_length,
        }
        return implementation(**kwargs)

    def apply_feature_wise_with_sampling(
        self,
        src_timestamps: Optional[np.ndarray],
        src_feature: Optional[np.ndarray],
        sampling_timestamps: np.ndarray,
        feature_idx: int,
    ) -> np.ndarray:
        """Applies the operator on a single feature with a sampling."""

        assert isinstance(self.operator, MovingCountOperator)
        implementation = self._implementation()

        if src_feature is not None:
            kwargs = {
                "evset_timestamps": src_timestamps,
                "window_length": self.operator.window_length,
                "sampling_timestamps": sampling_timestamps,
            }
            return implementation(**kwargs)
        else:
            # Sets the feature data as missing.
            empty_timestamps = np.empty((0,), dtype=np.float64)
            kwargs = {
                "evset_timestamps": empty_timestamps,
                "window_length": self.operator.window_length,
                "sampling_timestamps": sampling_timestamps,
            }
            return implementation(**kwargs)


implementation_lib.register_operator_implementation(
    MovingCountOperator, MovingCountNumpyImplementation
)
