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

from temporian.core.operators.select_index_values import SelectIndexValues
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.select_index_values import (
    SelectIndexValuesNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class SelectIndexValuesOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        evset = event_set(
            timestamps=[1, 2, 3],
            features={
                "a": [1.0, 2.0, 3.0],
                "b": [5, 6, 7],
                "c": ["A", "A", "B"],
            },
            indexes=["c"],
        )
        node = evset.node()

        expected_output = event_set(
            timestamps=[1, 2],
            features={
                "a": [1.0, 2.0],
                "b": [5, 6],
                "c": ["A", "A"],
            },
            indexes=["c"],
        )

        # Run op
        op = SelectIndexValues(input=node, keys=[("A",)])
        instance = SelectIndexValuesNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_many_indexes_many_keys_change_order(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "b": [5, 6, 7, 8, 9, 10],
                "c": ["A", "A", "B", "B", "C", "C"],
                "d": [1, 2, 1, 2, 1, 2],
            },
            indexes=["c", "d"],
        )
        node = evset.node()

        expected_output = event_set(
            timestamps=[5, 1, 4],
            features={
                "a": [5.0, 1.0, 4.0],
                "b": [9, 5, 8],
                "c": ["C", "A", "B"],
                "d": [1, 1, 2],
            },
            indexes=["c", "d"],
        )

        # Run op
        op = SelectIndexValues(input=node, keys=[("C", 1), ("A", 1), ("B", 2)])
        instance = SelectIndexValuesNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
