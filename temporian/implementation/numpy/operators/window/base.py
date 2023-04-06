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
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
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

        dst_event = NumpyEvent(data={}, sampling=sampling.sampling)

        # For each index
        for index, src_features in event.data.items():
            dst_features = []
            dst_event.data[index] = dst_features
            src_timestamps = event.sampling.data[index]
            sampling_timestamps = sampling.sampling.data[index]

            self._compute(
                src_timestamps, src_features, sampling_timestamps, dst_features
            )

        return {"event": dst_event}

    @abstractmethod
    def _implementation(self) -> Any:
        pass

    def _compute(
        self,
        src_timestamps: np.ndarray,
        src_features: List[NumpyFeature],
        sampling_timestamps: np.ndarray,
        dst_features: List[NumpyFeature],
    ):
        implementation = self._implementation()
        for src_ts in src_features:
            args = {
                "event_timestamps": src_timestamps,
                "event_values": src_ts.data,
                "window_length": self.operator.window_length(),
            }
            if self.operator.has_sampling():
                args["sampling_timestamps"] = sampling_timestamps
            dst_feature = implementation(**args)
            dst_features.append(NumpyFeature(src_ts.name, dst_feature))
