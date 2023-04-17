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
from typing import Dict
from abc import ABC, abstractmethod

from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.feature import NumpyFeature
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseArithmeticScalarNumpyImplementation(OperatorImplementation, ABC):
    def __init__(self, operator: BaseArithmeticScalarOperator) -> None:
        super().__init__(operator)

    @abstractmethod
    def _do_operation(self, feature: NumpyFeature, value: any) -> np.ndarray:
        """
        Perform the actual arithmetic operation corresponding to the subclass
        """

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        """Apply the corresponding arithmetic operation between an event and a
        scalar.

        Args:
            event: event to perform operation to.

        Returns:
            Arithmetic of the event and the valye according to the operator.
        """
        value = self._operator.attributes["value"]

        output = NumpyEvent(data={}, sampling=event.sampling)

        prefix = self._operator.prefix

        for index, features in event.data.items():
            output.data[index] = [
                NumpyFeature(
                    name=f"{prefix}_{feature.name}_{value}",
                    data=self._do_operation(feature, value),
                )
                for feature in features
            ]

        return {"event": output}
