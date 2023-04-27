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
from typing import Dict
from abc import ABC, abstractmethod

import numpy as np

from temporian.core.operators.arithmetic.base import BaseArithmeticOperator
from temporian.implementation.numpy.data.event import IndexData
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseArithmeticNumpyImplementation(OperatorImplementation, ABC):
    def __init__(self, operator: BaseArithmeticOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, BaseArithmeticOperator)

    @abstractmethod
    def _do_operation(
        self, event_1_feature: np.ndarray, event_2_feature: np.ndarray
    ) -> np.ndarray:
        """Performs the arithmetic operation corresponding to the subclass."""

    def __call__(
        self, event_1: NumpyEvent, event_2: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        """Applies the corresponding arithmetic operation between two events.

        Args:
            event_1: First event.
            event_2: Second event.

        Returns:
            Result of the operation.

        Raises:
            ValueError: If sampling of both events is not equal.
        """

        if event_1.feature_count != event_2.feature_count:
            raise ValueError(
                "Both events must have the same number of features."
            )

        # gather operator outputs
        prefix = self._operator.prefix

        # create destination event
        dst_feature_names = [
            f"{prefix}_{feature_name_1}_{feature_name_2}"
            for feature_name_1, feature_name_2 in zip(
                event_1.feature_names, event_2.feature_names
            )
        ]
        dst_event = NumpyEvent(
            data={},
            feature_names=dst_feature_names,
            index_names=event_1.index_names,
            is_unix_timestamp=event_1.is_unix_timestamp,
        )
        for index_key, index_data in event_1.iterindex():
            # initialize destination index data
            dst_event[index_key] = IndexData([], index_data.timestamps)

            # iterate over index key features
            event_1_features = index_data.features
            event_2_features = event_2[index_key].features
            for event_1_feature, event_2_feature in zip(
                event_1_features, event_2_features
            ):
                # check both features have the same dtype
                if event_1_feature.dtype.type != event_2_feature.dtype.type:
                    raise ValueError(
                        "Both features must have the same dtype."
                        f" event_1_feature: {event_1_feature} has dtype "
                        f"{event_1_feature.dtype}, event_2_feature: "
                        f"{event_2_feature} has dtype {event_2_feature.dtype}."
                    )

                result = self._do_operation(event_1_feature, event_2_feature)
                dst_event[index_key].features.append(result)

        return {"event": dst_event}
