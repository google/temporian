import numpy as np

from temporian.core.operators.sum import Resolution
from temporian.core.operators.multiply import MultiplyOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature


class MultiplyNumpyImplementation:
    def __init__(self, operator: MultiplyOperator) -> None:
        super().__init__()
        if not isinstance(operator, MultiplyOperator):
            raise TypeError(
                f"operator must be of type {MultiplyOperator}, not"
                f" {type(operator)}."
            )
        self.operator = operator

    def __call__(self, event_1: NumpyEvent, event_2: NumpyEvent) -> NumpyEvent:
        """Multiply two Events.

        Args:
            event_1: First Event.
            event_2: Second Event.
            resolution: Resolution of the output Event. PER_FEATURE_IDX multiplication is done feature index wise. PER_FEATURE_NAME multiplication is done feature name wise.

        Returns:
            Multiplication of the two Events according to resolution.

        Raises:
            ValueError: If sampling of both events is not equal.
            NotImplementedError: If resolution is PER_FEATURE_NAME.
        """
        if event_1.sampling != event_2.sampling:
            raise ValueError("Sampling of both events must be equal.")

        output = NumpyEvent(data={}, sampling=event_1.sampling)

        resolution = self.operator.attributes()["resolution"]

        if resolution == Resolution.PER_FEATURE_IDX:
            for event_1_index, event_1_features in event_1.data.items():
                output.data[event_1_index] = []

                for i in range(len(event_1_features)):
                    event_1_feature = event_1.data[event_1_index][i]
                    event_2_feature = event_2.data[event_1_index][i]

                    output.data[event_1_index].append(
                        NumpyFeature(
                            name=f"mult_{event_1_feature.name}_{event_2_feature.name}",
                            data=event_1_feature.data * event_2_feature.data,
                        )
                    )
        elif resolution == Resolution.PER_FEATURE_NAME:
            raise NotImplementedError(
                "PER_FEATURE_NAME resolution is not implemented yet."
            )
        else:
            raise ValueError(
                f"Resolution {resolution} is not supported. Choose from"
                f" {Resolution.PER_FEATURE_IDX} and"
                f" {Resolution.PER_FEATURE_NAME}."
            )

        return {"event": output}
