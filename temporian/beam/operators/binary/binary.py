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

import numpy as np
from abc import abstractmethod
from typing import Dict, Type, Optional
from temporian.core.operators.binary.base import BaseBinaryOperator

from temporian.implementation.numpy.operators.binary.base import (
    BaseBinaryNumpyImplementation,
)
from temporian.beam.operators.base import (
    BeamOperatorImplementation,
    beam_eventset_map_with_sampling,
)
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
    BeamIndexKey,
    FeatureItemValue,
)
from temporian.beam import implementation_lib

from temporian.implementation.numpy.operators.binary.arithmetic import (
    AddNumpyImplementation,
    SubtractNumpyImplementation,
    MultiplyNumpyImplementation,
    DivideNumpyImplementation,
    FloorDivNumpyImplementation,
    ModuloNumpyImplementation,
    PowerNumpyImplementation,
)

from temporian.implementation.numpy.operators.binary.relational import (
    EqualNumpyImplementation,
    GreaterEqualNumpyImplementation,
    GreaterNumpyImplementation,
    LessEqualNumpyImplementation,
    LessNumpyImplementation,
    NotEqualNumpyImplementation,
)

from temporian.implementation.numpy.operators.binary.logical import (
    LogicalAndNumpyImplementation,
    LogicalOrNumpyImplementation,
    LogicalXorNumpyImplementation,
)

from temporian.core.operators.binary.arithmetic import (
    AddOperator,
    SubtractOperator,
    MultiplyOperator,
    DivideOperator,
    FloorDivOperator,
    ModuloOperator,
    PowerOperator,
)

from temporian.core.operators.binary.relational import (
    EqualOperator,
    NotEqualOperator,
    GreaterOperator,
    GreaterEqualOperator,
    LessOperator,
    LessEqualOperator,
)

from temporian.core.operators.binary.logical import (
    LogicalAndOperator,
    LogicalOrOperator,
    LogicalXorOperator,
)


class BaseBinaryBeamImplementation(BeamOperatorImplementation):
    def __init__(self, operator: BaseBinaryOperator):
        super().__init__(operator)
        assert isinstance(operator, BaseBinaryOperator)

    @abstractmethod
    def _implementation(self) -> Type[BaseBinaryNumpyImplementation]:
        pass

    def call(
        self, input_1: BeamEventSet, input_2: BeamEventSet
    ) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, BaseBinaryOperator)
        numpy_implementation = self._implementation()(self.operator)

        input_1_dtypes = self.operator.inputs["input_1"].schema.feature_dtypes()

        def apply(
            index: BeamIndexKey,
            input_1: Optional[FeatureItemValue],
            input_2: FeatureItemValue,
            feature_idx: int,
        ) -> FeatureItem:
            # Binary operators only apply on features with similar sampling i.e.
            # the index and timestamps are the same.
            assert input_1 is not None

            input_1_timestamps, input_1_values = input_1
            _, input_2_values = input_2

            # Binary operators only apply on eventset with one feature.
            assert input_1_values is not None
            assert input_2_values is not None

            output_values = numpy_implementation._do_operation(
                evset_1_feature=input_1_values,
                evset_2_feature=input_2_values,
                dtype=input_1_dtypes[feature_idx],
            )
            return index, (input_1_timestamps, output_values)

        output = beam_eventset_map_with_sampling(
            input_1,
            input_2,
            name=f"{self.operator}",
            fn=apply,
        )
        return {"output": output}


def build_beam_imp_class(numpy_implementation):
    class BeamImplementation(BaseBinaryBeamImplementation):
        def _implementation(self) -> Type[BaseBinaryNumpyImplementation]:
            return numpy_implementation

    return BeamImplementation


for operator, numpy_implementation in [
    (AddOperator, AddNumpyImplementation),
    (SubtractOperator, SubtractNumpyImplementation),
    (MultiplyOperator, MultiplyNumpyImplementation),
    (DivideOperator, DivideNumpyImplementation),
    (FloorDivOperator, FloorDivNumpyImplementation),
    (ModuloOperator, ModuloNumpyImplementation),
    (PowerOperator, PowerNumpyImplementation),
    (EqualOperator, EqualNumpyImplementation),
    (NotEqualOperator, NotEqualNumpyImplementation),
    (GreaterOperator, GreaterNumpyImplementation),
    (GreaterEqualOperator, GreaterEqualNumpyImplementation),
    (LessOperator, LessNumpyImplementation),
    (LessEqualOperator, LessEqualNumpyImplementation),
    (LogicalAndOperator, LogicalAndNumpyImplementation),
    (LogicalOrOperator, LogicalOrNumpyImplementation),
    (LogicalXorOperator, LogicalXorNumpyImplementation),
]:
    implementation_lib.register_operator_implementation(
        operator, build_beam_imp_class(numpy_implementation)
    )
