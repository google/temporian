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

"""Arithmetic Negation Scalar Operator"""
from typing import List

from temporian.core import operator_lib
from temporian.core.data import dtype as dtype_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class NegateOperator(BaseArithmeticScalarOperator):
    """
    Negates the event features.
    """

    def __init__(
        self,
        event: Event,
        value: int = -1,
        is_value_first: bool = False,
    ):
        super().__init__(event, value, is_value_first)

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "NEGATE"

    # this will be ignored because we will override output feature name
    @property
    def prefix(self) -> str:
        return ""

    # overriding feature name to be the same as the input feature
    def output_feature_name(self, feature: Feature) -> str:
        return feature.name

    # overriding checking for feature dtype to be the same as value dtype
    @property
    def ignore_value_dtype_checking(self) -> bool:
        return True

    @property
    def supported_value_dtypes(self) -> List[dtype_lib.DType]:
        return [
            dtype_lib.INT32,
            dtype_lib.INT64,
        ]


operator_lib.register_operator(NegateOperator)


def negate(
    event: Event,
) -> Event:
    """
    Negates the event features.

    Args:
        event: Event to negate.

    Returns:
        Negated event.
    """
    return NegateOperator(
        event=event,
        value=-1,
    ).outputs["event"]
