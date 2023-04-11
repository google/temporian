from typing import Dict

import numpy as np

from temporian.core.operators.equal import EqualOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.feature import NumpyFeature
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class EqualNumpyImplementation(OperatorImplementation):
    """New event with booleans where features are equal to a value."""

    def __init__(self, operator: EqualOperator) -> None:
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        value = self.operator.attributes["value"]

        output_event = NumpyEvent(data={}, sampling=event.sampling)

        for index_value, features in event.data.items():
            equal_features = []

            for feature in features:
                # don't compare strings with numbers
                if (
                    isinstance(value, str)
                    and feature.data.dtype.type != np.str_
                    or not isinstance(value, str)
                    and feature.data.dtype.type == np.str_
                ):
                    equal_data = np.full(feature.data.shape, False)
                else:
                    equal_data = np.equal(feature.data, value)

                equal_features.append(
                    NumpyFeature(f"{feature.name}_equal_{value}", equal_data)
                )

            output_event.data[index_value] = equal_features

        return {"event": output_event}


implementation_lib.register_operator_implementation(
    EqualOperator, EqualNumpyImplementation
)
