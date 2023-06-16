from temporian.core.operators.window.moving_sum import (
    MovingSumOperator as CurrentOperator,
)
from temporian.beam import implementation_lib
from temporian.implementation.numpy.operators.window.moving_sum import (
    MovingSumNumpyImplementation as CurrentOperatorImplementation,
)
from temporian.beam.operators.base import BeamOperatorImplementation
from typing import Dict
from temporian.beam.io import BeamEventSet, PColBeamEventSet
import apache_beam as beam
from temporian.implementation.numpy.operators.base import OperatorImplementation


def _run_item(pipe: BeamEventSet, imp: OperatorImplementation):
    indexes, (timestamps, input_values) = pipe
    output_values = imp.apply_feature_wise(
        src_timestamps=timestamps,
        src_feature=input_values,
    )
    return indexes, (timestamps, output_values)


class MovingSumBeamImplementation(BeamOperatorImplementation):
    def call(self, input: PColBeamEventSet) -> Dict[str, PColBeamEventSet]:
        assert isinstance(self.operator, CurrentOperator)

        assert not self.operator.has_sampling

        numpy_implementation = CurrentOperatorImplementation(self.operator)

        output = input | f"Apply operator {self.operator}" >> beam.Map(
            _run_item, numpy_implementation
        )

        return {"output": output}


implementation_lib.register_operator_implementation(
    CurrentOperator, MovingSumBeamImplementation
)
