# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Dict, Type
from temporian.core.operators.scalar.base import BaseScalarOperator

from temporian.implementation.numpy.operators.scalar.base import (
    BaseScalarNumpyImplementation,
)
from temporian.beam.operators.base import (
    BeamOperatorImplementation,
    beam_eventset_map,
)
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
)
from temporian.beam import implementation_lib

from temporian.implementation.numpy.operators.scalar.arithmetic_scalar import (
    AddScalarNumpyImplementation,
    SubtractScalarNumpyImplementation,
    MultiplyScalarNumpyImplementation,
    DivideScalarNumpyImplementation,
    FloorDivideScalarNumpyImplementation,
    ModuloScalarNumpyImplementation,
    PowerScalarNumpyImplementation,
)

from temporian.implementation.numpy.operators.scalar.relational_scalar import (
    EqualScalarNumpyImplementation,
    NotEqualScalarNumpyImplementation,
    GreaterEqualScalarNumpyImplementation,
    GreaterScalarNumpyImplementation,
    LessEqualScalarNumpyImplementation,
    LessScalarNumpyImplementation,
)

from temporian.core.operators.scalar.arithmetic_scalar import (
    AddScalarOperator,
    SubtractScalarOperator,
    MultiplyScalarOperator,
    DivideScalarOperator,
    FloorDivScalarOperator,
    ModuloScalarOperator,
    PowerScalarOperator,
)

from temporian.core.operators.scalar.relational_scalar import (
    EqualScalarOperator,
    NotEqualScalarOperator,
    GreaterEqualScalarOperator,
    GreaterScalarOperator,
    LessEqualScalarOperator,
    LessScalarOperator,
)


class BaseScalarBeamImplementation(BeamOperatorImplementation):
    def __init__(self, operator: BaseScalarOperator):
        super().__init__(operator)
        assert isinstance(operator, BaseScalarOperator)

    @abstractmethod
    def _implementation(self) -> Type[BaseScalarNumpyImplementation]:
        pass

    def call(self, input: BeamEventSet) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, BaseScalarOperator)
        numpy_implementation = self._implementation()(self.operator)
        value = self.operator.value

        input_dtypes = self.operator.inputs["input"].schema.feature_dtypes()

        def apply(
            item: FeatureItem,
            feature_idx: int,
        ) -> FeatureItem:
            indexes, (timestamps, input_values) = item
            assert input_values is not None
            output_values = numpy_implementation._do_operation(
                feature=input_values,
                value=value,
                dtype=input_dtypes[feature_idx],
            )
            return indexes, (timestamps, output_values)

        output = beam_eventset_map(
            input,
            name=f"{self.operator}",
            fn=apply,
        )
        return {"output": output}


def build_beam_imp_class(numpy_implementation):
    class BeamImplementation(BaseScalarBeamImplementation):
        def _implementation(self) -> Type[BaseScalarNumpyImplementation]:
            return numpy_implementation

    return BeamImplementation


for operator, numpy_implementation in [
    (EqualScalarOperator, EqualScalarNumpyImplementation),
    (NotEqualScalarOperator, NotEqualScalarNumpyImplementation),
    (GreaterEqualScalarOperator, GreaterEqualScalarNumpyImplementation),
    (LessEqualScalarOperator, LessEqualScalarNumpyImplementation),
    (GreaterScalarOperator, GreaterScalarNumpyImplementation),
    (LessScalarOperator, LessScalarNumpyImplementation),
    (AddScalarOperator, AddScalarNumpyImplementation),
    (SubtractScalarOperator, SubtractScalarNumpyImplementation),
    (MultiplyScalarOperator, MultiplyScalarNumpyImplementation),
    (DivideScalarOperator, DivideScalarNumpyImplementation),
    (FloorDivScalarOperator, FloorDivideScalarNumpyImplementation),
    (ModuloScalarOperator, ModuloScalarNumpyImplementation),
    (PowerScalarOperator, PowerScalarNumpyImplementation),
]:
    implementation_lib.register_operator_implementation(
        operator, build_beam_imp_class(numpy_implementation)
    )
