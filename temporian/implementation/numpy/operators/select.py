from typing import Dict

from temporian.core.operators.select import SelectOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class SelectNumpyImplementation(OperatorImplementation):
    """Select a subset of features from an event."""

    def __init__(self, operator: SelectOperator) -> None:
        super().__init__(operator)

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        feature_names = self.operator.attributes()["feature_names"]

        output_event = NumpyEvent(
            {
                index_value: [
                    feature
                    for feature in features
                    if feature.name in feature_names
                ]
                for index_value, features in event.data.items()
            },
            event.sampling,
        )
        return {"event": output_event}


implementation_lib.register_operator_implementation(
    SelectOperator, SelectNumpyImplementation
)
