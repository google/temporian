from typing import Dict, List

from temporian.implementation.numpy.data.event import NumpyEvent


class NumpySelectOperator:
    """Select a subset of features from an event."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    def __call__(self, event: NumpyEvent) -> Dict[str, NumpyEvent]:
        output_event = NumpyEvent(
            {
                index_value: [
                    feature
                    for feature in features
                    if feature.name in self.feature_names
                ]
                for index_value, features in event.data.items()
            },
            event.sampling,
        )
        return {"event": output_event}
