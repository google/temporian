from temporian.core.operators.select import (
    SelectOperator as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.implementation.numpy.operators.select import (
    SelectNumpyImplementation as CurrentOperatorImplementation,
)
from temporian.beam.operators.base import BeamOperatorImplementation
from typing import Dict
from temporian.beam.io import BeamEventSet, PColBeamEventSet
import apache_beam as beam
from temporian.implementation.numpy.operators.base import OperatorImplementation


class SelectBeamImplementation(BeamOperatorImplementation):
    def call(self, input: PColBeamEventSet) -> Dict[str, PColBeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        assert not self.operator.has_sampling

        # TODO
        return {}


implementation_lib.register_operator_implementation(
    CurrentOperator, SelectBeamImplementation
)
