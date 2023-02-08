from typing import Dict, List, Tuple

import numpy as np

from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.implementation.numpy.data.sampling import NumpySampling


class NumpyFeature:
    def __init__(
        self, name: str, sampling: NumpySampling, data: np.ndarray
    ) -> None:
        if len(data.shape) > 1:
            raise ValueError(
                "NumpyFeatures can only be created from flat arrays. Passed"
                f" input's shape: {len(data.shape)}"
            )

        self.name = name
        self.sampling = sampling
        self.data = data
        self.dtype = self.data.dtype

    def schema(self) -> Feature:
        return Feature(
            name=self.name, dtype=self.dtype, sampling=self.sampling.names
        )

    def __repr__(self) -> str:
        return f"{self.name}: {self.data.__repr__()}"


class NumpyEvent:
    def __init__(
        self,
        data: Dict[Tuple, List[NumpyFeature]],
        sampling: NumpySampling,
    ) -> None:
        self.data = data
        self.sampling = sampling

    def schema(self) -> Event:
        return Event(
            features=[
                feature.schema() for feature in list(self.data.values())[0]
            ],
            sampling=self.sampling.names,
        )

    def __repr__(self) -> str:
        return self.data.__repr__()
