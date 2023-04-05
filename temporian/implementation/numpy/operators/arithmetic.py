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

from temporian.core.operators.arithmetic import ArithmeticOperation
from temporian.core.operators.arithmetic import ArithmeticOperator
from temporian.core.operators.arithmetic import Resolution
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class ArithmeticNumpyImplementation(OperatorImplementation):
    """Numpy implementation of the arithmetic operator."""

    def __init__(self, operator: ArithmeticOperator) -> None:
        super().__init__(operator)

    def __call__(
        self, event_1: NumpyEvent, event_2: NumpyEvent
    ) -> Dict[str, NumpyEvent]:
        """Sums two NumpyEvents.

        Args:
            event_1: First event.
            event_2: Second event.

        Returns:
            Result of the arithmetic operation of the two Events according to
            resolution and operator.

        Raises:
            ValueError: If sampling of both events is not equal.
            NotImplementedError: If resolution is PER_FEATURE_NAME.
        """
        resolution = self.operator.attributes()["resolution"]
        operation = self.operator.attributes()["operation"]

        if event_1.sampling is not event_2.sampling:
            raise ValueError("Sampling of both events must be equal.")

        if event_1.feature_count != event_2.feature_count:
            raise ValueError(
                "Both events must have the same number of features."
            )

        if not ArithmeticOperation.is_valid(operation):
            raise ValueError(f"Unknown operation: {operation}.")

        output = NumpyEvent(data={}, sampling=event_1.sampling)

        if resolution == Resolution.PER_FEATURE_NAME:
            raise NotImplementedError(
                "PER_FEATURE_NAME is not implemented yet."
            )

        prefix = ArithmeticOperation.prefix(operation)

        for event_index, event_1_features in event_1.data.items():
            output.data[event_index] = []

            event_2_features = event_2.data[event_index]

            for i, event_1_feature in enumerate(event_1_features):
                event_2_feature = event_2_features[i]

                # check both features have the same dtype
                if event_1_feature.dtype != event_2_feature.dtype:
                    raise ValueError(
                        "Both features must have the same dtype."
                        f" event_1_feature: {event_1_feature} has dtype "
                        f"{event_1_feature.dtype}, event_2_feature: "
                        f"{event_2_feature} has dtype {event_2_feature.dtype}."
                    )

                data = None

                if operation == ArithmeticOperation.ADDITION:
                    data = event_1_feature.data + event_2_feature.data
                elif operation == ArithmeticOperation.SUBTRACTION:
                    data = event_1_feature.data - event_2_feature.data
                elif operation == ArithmeticOperation.MULTIPLICATION:
                    data = event_1_feature.data * event_2_feature.data
                elif operation == ArithmeticOperation.DIVISION:
                    data = event_1_feature.data / event_2_feature.data

                output.data[event_index].append(
                    NumpyFeature(
                        name=f"{prefix}_{event_1_feature.name}_{event_2_feature.name}",
                        data=data,
                    )
                )

        return {"event": output}


implementation_lib.register_operator_implementation(
    ArithmeticOperator, ArithmeticNumpyImplementation
)
