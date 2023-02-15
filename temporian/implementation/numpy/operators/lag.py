import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.core.data.duration import Duration


class NumpyLagOperator:
    # TODO: Check Duration = None
    def __init__(self, duration: Duration = None) -> None:
        super().__init__()
        self.duration = duration

    def __call__(self, event: NumpyEvent) -> NumpyEvent:
        output = NumpyEvent(data={}, sampling=event.sampling)

        for index in event.data.keys():
            # Create matrix where False indicates that lag has been reached
            # it is inverted for the next step.
            mask = (
                event.sampling.data[index][:, np.newaxis]
                < event.sampling.data[index][np.newaxis, :] + self.duration
            )
            # We have to find the last False in order to get the first element
            # that meet the lag. We do this by finding the first True with
            # argmax and subtracting 1. The matrix is inverted because if
            # we want to find the last True, argmax will return the first one.
            last_true_idx = np.argmax(mask, axis=1)
            lag_index = last_true_idx - 1
            output.data[index] = []

            for feature in event.data[index]:
                lagged_feature = NumpyFeature(
                    name=f"lag_{feature.name}", data=feature.data[lag_index]
                )
                lagged_feature.data[lag_index == -1] = np.nan
                output.data[index].append(lagged_feature)

        return {"event": output}
