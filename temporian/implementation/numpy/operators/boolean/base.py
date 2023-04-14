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
        self, event_1: NumpyEvent, event_2: NumpyEvent = None
    ) -> Dict[str, NumpyEvent]:
        # if operator is a Scalar Operator, value is the scalar
        if isinstance(self.operator, BaseBooleanScalarOperator):
            value = self.operator.attributes["value"]

        if event_2 and event_1.sampling != event_2.sampling:
            raise ValueError(
                "Event 1 and event 2 must have same sampling. Current"
                f" samplings: {event_1.sampling}, {event_2.sampling}"
            )

        # this is needed for .call() to work. It says that event_2 has different
        # sampling than output.
        if event_2:
            event_2.sampling = event_1.sampling

        output_event = NumpyEvent(data={}, sampling=event_1.sampling)

        for index_value, features in event_1.data.items():
            # if operator is Feature Operator, value is the feature array
            if isinstance(self.operator, BaseBooleanFeatureOperator):
                # get first and only feature in event_2
                value = event_2.data[index_value][0].data

            equal_features = [
                NumpyFeature(
                    self.operator.feature_name(feature),
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
