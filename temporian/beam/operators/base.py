from abc import ABC
from temporian.core.operators.base import Operator
from abc import ABC, abstractmethod
from temporian.beam.io import BeamEventSet, PColBeamEventSet
from typing import Dict
import apache_beam as beam


class BeamOperatorImplementation(ABC):
    def __init__(self, operator: Operator):
        assert operator is not None
        self._operator = operator

    @property
    def operator(self):
        return self._operator

    @abstractmethod
    def call(self, **inputs: PColBeamEventSet) -> Dict[str, PColBeamEventSet]:
        pass

    def __call__(
        self, **inputs: PColBeamEventSet
    ) -> Dict[str, PColBeamEventSet]:
        outputs = self.call(**inputs)
        return outputs
