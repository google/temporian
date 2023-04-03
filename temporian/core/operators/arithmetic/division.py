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

"""Division Operator"""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.arithmetic.base import BaseArithmeticOperator
from temporian.core.operators.arithmetic import Resolution


class DivisionOperator(BaseArithmeticOperator):
    """
    Divide first event by second one
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "DIVISION"

    @property
    def prefix(self) -> str:
        return "div"


operator_lib.register_operator(DivisionOperator)


def divide(
    numerator: Event,
    denominator: Event,
    resolution: Resolution = Resolution.PER_FEATURE_IDX,
) -> Event:
    """
    Divide two events.

    Args:
        numerator: Numerator event
        denominator: Denominator event
        resolution: If resolution is Resolution.PER_FEATURE_IDX each feature
            will be divide index wise. If resolution is
            Resolution.PER_FEATURE_NAME each feature of numerator will be
            divide with the feature in denominator with the same name.
            Defaults to Resolution.PER_FEATURE_IDX.

    Returns:
        Event: Division of numerator and denominator according to resolution.
    """
    return DivisionOperator(
        event_1=numerator,
        event_2=denominator,
        resolution=resolution,
    ).outputs()["event"]
