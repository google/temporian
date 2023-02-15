import numpy as np

from typing import Optional

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.core.data.duration import Duration


class NumpyLagOperator:
    # TODO: Check Duration = None
    def __init__(
        self, duration: Optional[Duration] = None, period: Optional[int] = None
    ) -> None:
        super().__init__()
        self.duration = duration
        self.period = period

    def __call__(self, event: NumpyEvent) -> NumpyEvent:
        output = NumpyEvent(data={}, sampling=event.sampling)

        for index in event.data.keys():
            mask = (
                event.sampling.data[index][:, np.newaxis]
                == event.sampling.data[index][np.newaxis, :] + self.duration
            )
            mask_i, mask_j = np.nonzero(mask)

            output.data[index] = []

            for feature in event.data[index]:
                lagged_feature = NumpyFeature(
                    name=f"lag_{feature.name}",
                    data=np.full(feature.data.shape, np.nan),
                )
                lagged_feature.data[mask_i] = feature.data[mask_j]
                output.data[index].append(lagged_feature)

        return {"event": output}
