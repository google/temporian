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
from typing import Dict, Union
from abc import ABC, abstractmethod

import numpy as np

from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.event_set import IndexData
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseArithmeticScalarNumpyImplementation(OperatorImplementation, ABC):
    def __init__(self, operator: BaseArithmeticScalarOperator) -> None:
        super().__init__(operator)

    @abstractmethod
    def _do_operation(
        self, feature: np.ndarray, value: Union[float, int, str, bool]
    ) -> np.ndarray:
        """Performs the arithmetic operation corresponding to the subclass."""

    def __call__(self, node: EventSet) -> Dict[str, EventSet]:
        """Applies the corresponding arithmetic operation between an event set
        and a scalar.

        Args:
            node: Event set to perform the operation to.

        Returns:
            Result of the operation.
        """
        dst_evset = EventSet(
            data={},
            feature_names=node.feature_names,
            index_names=node.index_names,
            is_unix_timestamp=node.is_unix_timestamp,
        )
        for index_key, index_data in node.iterindex():
            dst_evset[index_key] = IndexData(
                [
                    self._do_operation(feature, self._operator.value)
                    for feature in index_data.features
                ],
                index_data.timestamps,
            )

        return {"node": dst_evset}
