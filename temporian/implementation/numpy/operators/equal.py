from typing import Dict, Union

import numpy as np

from temporian.core.operators.boolean.equal_scalar import EqualScalarOperator
from temporian.core.operators.boolean.equal_feature import EqualFeatureOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.feature import NumpyFeature
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class EqualNumpyImplementation(OperatorImplementation):
    """New event with booleans where features are equal to a value."""

    def __init__(
        self, operator: Union[EqualScalarOperator, EqualFeatureOperator]
    ) -> None:
        assert isinstance(
            operator, (EqualScalarOperator, EqualFeatureOperator)
        ), (
            "Expected operator to be of type EqualScalarOperator or"
            f" EqualFeatureOperator, got {type(operator)}"
        )
        super().__init__(operator)

    def __call__(
        self, event: NumpyEvent, event_2: NumpyEvent = None
    ) -> Dict[str, NumpyEvent]:
        name = ""

        # if operator is EqualScalarOperator, value is the scalar
        if isinstance(self.operator, EqualScalarOperator):
            value = self.operator.attributes["value"]
            name = value

        # if operator is EqualFeatureOperator, value is the feature array
        if isinstance(self.operator, EqualFeatureOperator):
            value = event_2.data
            name = self.operator.inputs["event_2"]

        output_event = NumpyEvent(data={}, sampling=event.sampling)

        for index_value, features in event.data.items():
            equal_features = [
                NumpyFeature(
                    self.operator.feature_name(feature, name),
                    np.equal(feature.data, value),
                )
                for feature in features
            ]

            output_event.data[index_value] = equal_features

        return {"event": output_event}


implementation_lib.register_operator_implementation(
    EqualScalarOperator, EqualNumpyImplementation
)

implementation_lib.register_operator_implementation(
    EqualFeatureOperator, EqualNumpyImplementation
)
