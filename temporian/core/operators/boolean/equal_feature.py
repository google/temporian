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

"""Equal feature operator."""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.boolean.base_feature import (
    BaseBooleanFeatureOperator,
)


class EqualFeatureOperator(BaseBooleanFeatureOperator):
    """
    Boolean operator to compare if a feature is equal to a another feature.
    """

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "EQUAL_FEATURE"

    @property
    def operation_name(self) -> str:
        return "equal"


operator_lib.register_operator(EqualFeatureOperator)


def equal_feature(
    event_1: Event,
    event_2: Event,
) -> Event:
    """Equal feature..

    For each feature independently of event_1, returns a boolean feature that is
    True if the feature in event_1 is equal elementwise to the feature in
    event_2.


    Args:
        event_1: event to compare.
        event_2: event with only one feature for comparison.

    Returns:
        Event: event with the comparison result.

    Raises:
        ValueError: if event_2 has more than a single feature.
        ValueError: if event_1 and event_2 have different sampling.
        ValueError: if event_1 and event_2 have different dtypes.

    """
    return EqualFeatureOperator(event_1, event_2).outputs["event"]
