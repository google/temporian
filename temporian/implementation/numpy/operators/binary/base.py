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
from abc import abstractmethod

import numpy as np

from temporian.core.operators.binary.base import BaseBinaryOperator
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseBinaryNumpyImplementation(OperatorImplementation):
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
        """Applies the corresponding arithmetic operation between two EventSets.

        Args:
            input_1: First event set.
            input_2: Second event set.

        Returns:
            Result of the operation.

        Raises:
            ValueError: If sampling of both EventSets is not equal.
        """
        assert isinstance(self.operator, BaseBinaryOperator)
        output_schema = self.output_schema("output")

        if len(input_1.schema.features) != len(input_2.schema.features):
            raise ValueError(
                "Both EventSets must have the same number of features."
            )
        num_features = len(input_1.schema.features)

        # create destination event set
        dst_evset = EventSet(data={}, schema=output_schema)

        assert len(input_1.data) == len(input_2.data)

        for index_key, index_data in input_1.data.items():
            # iterate over index key features
            input_1_features = index_data.features
            input_2_features = input_2[index_key].features
            dst_features = []

            for feature_idx in range(num_features):
                input_1_feature = input_1_features[feature_idx]
                input_2_feature = input_2_features[feature_idx]
                assert input_1_feature.dtype.type == input_2_feature.dtype.type

                result = self._do_operation(input_1_feature, input_2_feature)
                dst_features.append(result)

            dst_evset[index_key] = IndexData(
                features=dst_features,
                timestamps=index_data.timestamps,
                schema=output_schema,
            )

        return {"output": dst_evset}
