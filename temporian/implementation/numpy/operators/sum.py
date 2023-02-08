import numpy as np

from temporian.core.operators.sum import Resolution
from temporian.implementation.numpy.data.event import NumpyEvent


class NumpySumOperator:
    def __init__(
        self, resolution: Resolution = Resolution.PER_FEATURE_IDX
    ) -> None:
        super().__init__()
        self.resolution = resolution

    def __call__(self, event_1: NumpyEvent, event_2: NumpyEvent) -> NumpyEvent:
        """Sum two Events.

        Args:
            event_1: First Event.
            event_2: Second Event.
            resolution: Resolution of the output Event. PER_FEATURE_IDX sum is done feature index wise. PER_FEATURE_NAME sum is done feature name wise.

        Returns:
            Sum of the two Events according to resolution.

        Raises:
            IndexError: If index of both events is not equal.
            NotImplementedError: If resolution is PER_FEATURE_NAME.
        """
        if event_1.data.keys() != event_2.data.keys():
            raise IndexError("Index of both events must be equal.")

        output = NumpyEvent(data={}, sampling=event_1.sampling)

        for event_index, event_index_array in event_1.data.items():
            output.data[event_index] = []

            for i in range(len(event_index_array)):
                event_1_feature = event_1.data[event_index][i]
                event_2_feature = event_2.data[event_index][i]

                output.data[event_index].append(
                    (
                        f"sum_{event_1_feature.name}_{event_2_feature.name}",
                        event_1_feature.data + event_2_feature.data,
                    )
                )

        return {"event": output}
