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

"""Equal scalar operator."""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.boolean.base_scalar import (
    BaseBooleanScalarOperator,
)


class EqualScalarOperator(BaseBooleanScalarOperator):
    """
    Boolean operator to compare if a feature is equal to a scalar.
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "EQUAL_SCALAR"

    @property
    def operation_name(self) -> str:
        return "equal"


operator_lib.register_operator(EqualScalarOperator)


def equal_scalar(
    event: Event,
    value: any,
) -> Event:
    """Equal scalar.

    For each feature independently, returns a boolean feature that is True if
    the feature is equal to the scalar value, and False otherwise.

    Args:
        event: event to compare.
        value: scalar value to compare to.

    Returns:
        Event: event with the comparison result.

    Raises:
        ValueError: if the event feature dtypes are different from the value
            dtype.
    """
    return EqualScalarOperator(event, value).outputs["event"]
