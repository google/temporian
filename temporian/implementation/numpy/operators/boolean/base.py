from typing import Dict, Union
from abc import abstractmethod

import numpy as np

from temporian.core.operators.boolean.base_scalar import (
    BaseBooleanScalarOperator,
)
from temporian.core.operators.boolean.base_feature import (
    BaseBooleanFeatureOperator,
)
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.feature import NumpyFeature
from temporian.implementation.numpy.operators.base import OperatorImplementation


class BaseBooleanNumpyImplementation(OperatorImplementation):
    """Abstract base class to implement common logic of numpy implementation of
    boolean operators."""

    def __init__(
        self,
        operator: Union[BaseBooleanScalarOperator, BaseBooleanFeatureOperator],
    ) -> None:
        assert isinstance(
            operator, (BaseBooleanScalarOperator, BaseBooleanFeatureOperator)
        ), (
            "Expected operator to be of type BaseBooleanScalarOperator or"
            f" BaseBooleanFeatureOperator, got {type(operator)}"
        )
        super().__init__(operator)

    def __call__(
        self, event: NumpyEvent, event_2: NumpyEvent = None
    ) -> Dict[str, NumpyEvent]:
        name = ""

        # if operator is a Scalar Operator, value is the scalar
        if isinstance(self.operator, BaseBooleanScalarOperator):
            value = self.operator.attributes["value"]
            name = value

        # if operator is Feature Operator, value is the feature array
        if isinstance(self.operator, BaseBooleanFeatureOperator):
            value = event_2.data
            name = self.operator.inputs["event_2"]

        output_event = NumpyEvent(data={}, sampling=event.sampling)

        for index_value, features in event.data.items():
            equal_features = [
                NumpyFeature(
                    self.operator.feature_name(feature, name),
                    self.operation(feature.data, value),
                )
                for feature in features
            ]

            output_event.data[index_value] = equal_features

        return {"event": output_event}

    @abstractmethod
    def operation(
        self, feature_data: np.ndarray, value: Union[np.ndarray, any]
    ) -> np.ndarray:
        """Implements the boolean operation of the operator.

        Args:
            feature_data: Array of feature data.
            value: Value to compare the feature data to. Can be a scalar or
                numpy array.

        Returns:
            np.ndarray: Array of bools.
        """
