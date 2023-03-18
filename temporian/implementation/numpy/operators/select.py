from typing import Dict

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy import implementation_lib
from temporian.core.operators.select import SelectOperator


class NumpySelectOperator:
    """Select a subset of features from an event."""

    def __init__(self, op: SelectOperator) -> None:
        assert isinstance(op, SelectOperator)
        self._op = op

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        feature_names = self._op.attributes()["feature_names"]

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
    SelectOperator, NumpySelectOperator
)
