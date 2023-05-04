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

from temporian.core.operators.binary.base import BaseBinaryOperator
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseBinaryNumpyImplementation(OperatorImplementation, ABC):
    def __init__(self, operator: BaseBinaryOperator) -> None:
        super().__init__(operator)
        assert isinstance(operator, BaseBinaryOperator)

    @abstractmethod
    def _do_operation(
        self, evset_1_feature: np.ndarray, evset_2_feature: np.ndarray
    ) -> np.ndarray:
        """Performs the arithmetic operation corresponding to the subclass."""

    def __call__(
        self, input_1: EventSet, input_2: EventSet
    ) -> Dict[str, EventSet]:
        """Applies the corresponding arithmetic operation between two event
        sets.

        Args:
            input_1: First event set.
            input_2: Second event set.

        Returns:
            Result of the operation.

        Raises:
            ValueError: If sampling of both event sets is not equal.
        """

        if input_1.feature_count != input_2.feature_count:
            raise ValueError(
                "Both event sets must have the same number of features."
            )

        # gather operator outputs
        prefix = self._operator.prefix

        # create destination event set
        dst_feature_names = [
            f"{prefix}_{feature_name_1}_{feature_name_2}"
            for feature_name_1, feature_name_2 in zip(
                input_1.feature_names, input_2.feature_names
            )
        ]
        dst_evset = EventSet(
            data={},
            feature_names=dst_feature_names,
            index_names=input_1.index_names,
            is_unix_timestamp=input_1.is_unix_timestamp,
        )
        for index_key, index_data in input_1.iterindex():
            # initialize destination index data
            dst_evset[index_key] = IndexData([], index_data.timestamps)

            # iterate over index key features
            input_1_features = index_data.features
            input_2_features = input_2[index_key].features
            for input_1_feature, input_2_feature in zip(
                input_1_features, input_2_features
            ):
                # check both features have the same dtype
                if input_1_feature.dtype.type != input_2_feature.dtype.type:
                    raise ValueError(
                        "Both features must have the same dtype."
                        f" input_1_feature: {input_1_feature} has dtype "
                        f"{input_1_feature.dtype}, input_2_feature: "
                        f"{input_2_feature} has dtype {input_2_feature.dtype}."
                    )

                result = self._do_operation(input_1_feature, input_2_feature)
                dst_evset[index_key].features.append(result)

        return {"output": dst_evset}
