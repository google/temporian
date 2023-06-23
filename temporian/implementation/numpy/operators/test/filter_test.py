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

from absl.testing import absltest

from temporian.core.operators.filter import FilterOperator
from temporian.implementation.numpy.operators.filter import (
    FilterNumpyImplementation,
)
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class FilterOperatorTest(absltest.TestCase):
    """Filter operator test."""

    def test_base(self) -> None:
        input_data = event_set(
            timestamps=[1, 2, 3], features={"x": [4, 5, 6], "y": [7, 8, 9]}
        )
        input_condition = event_set(
            timestamps=[1, 2, 3],
            features={"c": [True, True, False]},
            same_sampling_as=input_data,
        )
        expected_result = event_set(
            timestamps=[1, 2], features={"x": [4, 5], "y": [7, 8]}
        )
        operator = FilterOperator(
            input=input_data.node(), condition=input_condition.node()
        )
        impl = FilterNumpyImplementation(operator)
        testOperatorAndImp(self, operator, impl)
        filtered_evset = impl.call(input=input_data, condition=input_condition)[
            "output"
        ]
        assertEqualEventSet(self, filtered_evset, expected_result)

    def test_index(self) -> None:
        input_data = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [4, 5, 6, 7], "y": [1, 1, 2, 2]},
            indexes=["y"],
        )
        input_condition = event_set(
            timestamps=[1, 2, 3, 4],
            features={"c": [True, True, True, False], "y": [1, 1, 2, 2]},
            indexes=["y"],
            same_sampling_as=input_data,
        )
        expected_result = event_set(
            timestamps=[1, 2, 3],
            features={"x": [4, 5, 6], "y": [1, 1, 2]},
            indexes=["y"],
        )
        operator = FilterOperator(
            input=input_data.node(), condition=input_condition.node()
        )
        impl = FilterNumpyImplementation(operator)
        testOperatorAndImp(self, operator, impl)
        filtered_evset = impl.call(input=input_data, condition=input_condition)[
            "output"
        ]
        assertEqualEventSet(self, filtered_evset, expected_result)


if __name__ == "__main__":
    absltest.main()
