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
from typing import Dict, Optional, List, Any

import numpy as np
from temporian.core.operators.window.base import BaseWindowOperator
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseWindowNumpyImplementation(OperatorImplementation):
    """Abstract base class to implement common logic of numpy implementation of
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

        dst_event_data = {}

        # For each index
        for index_key, index_data in event.data.items():
            dst_features = []
            sampling_timestamps = sampling.data[index_key].timestamps
            dst_event_data[index_key] = IndexData(
                dst_features, sampling_timestamps
            )
            self._compute(
                src_timestamps=index_data.timestamps,
                src_features=index_data.features,
                sampling_timestamps=sampling_timestamps,
                dst_features=dst_features,
            )

        return {
            "event": NumpyEvent(
                dst_event_data,
                feature_names=event.feature_names,
                index_names=sampling.index_names,
            )
        }

    @abstractmethod
    def _implementation(self) -> Any:
        pass

    def _compute(
        self,
        src_timestamps: np.ndarray,
        src_features: List[np.ndarray],
        sampling_timestamps: np.ndarray,
        dst_features: List[np.ndarray],
    ) -> None:
        implementation = self._implementation()
        for src_ts in src_features:
            args = {
                "event_timestamps": src_timestamps,
                "event_values": src_ts,
                "window_length": self.operator.window_length,
            }
            if self.operator.has_sampling:
                args["sampling_timestamps"] = sampling_timestamps
            dst_feature = implementation(**args)
            dst_features.append(dst_feature)
