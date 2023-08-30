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
        self.evset = event_set(
            timestamps=[1, 2, 3],
            features={
                "a": [1.0, 2.0, 3.0],
                "b": [5, 6, 7],
                "c": ["A", "A", "B"],
            },
            indexes=["c"],
        )
        self.node = self.evset.node()

    def test_basic(self):
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
        op = SelectIndexValues(input=self.node, keys=[("A",)])
        instance = SelectIndexValuesNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=self.evset)["output"]

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

    def test_single_index_key_value(self):
        expected_output = event_set(
            timestamps=[3],
            features={
                "a": [3.0],
                "b": [7],
                "c": ["B"],
            },
            indexes=["c"],
        )

        # Run op
        op = SelectIndexValues(input=self.node, keys="B")
        instance = SelectIndexValuesNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=self.evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_number(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            features={
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            indexes=["a"],
        )
        op = SelectIndexValues(input=evset.node(), number=3)
        instance = SelectIndexValuesNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        self.assertEqual(len(output.data.keys()), 3)

    def test_number_larger_than_total(self):
        evset = event_set(
            timestamps=[1, 2, 3],
            features={
                "a": [1, 2, 3],
            },
            indexes=["a"],
        )
        op = SelectIndexValues(input=evset.node(), number=4)
        instance = SelectIndexValuesNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        self.assertEqual(len(output.data.keys()), 3)

    def test_fraction(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            features={
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            indexes=["a"],
        )
        op = SelectIndexValues(input=evset.node(), fraction=0.7)
        instance = SelectIndexValuesNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        self.assertEqual(len(output.data.keys()), 7)

    def test_fraction_0(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            features={
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            indexes=["a"],
        )
        op = SelectIndexValues(input=evset.node(), fraction=0)
        instance = SelectIndexValuesNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        self.assertEqual(len(output.data.keys()), 0)

    def test_fraction_1(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            features={
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            indexes=["a"],
        )
        op = SelectIndexValues(input=evset.node(), fraction=1)
        instance = SelectIndexValuesNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        self.assertEqual(len(output.data.keys()), 10)

    def test_wrong_index_key(self):
        with self.assertRaisesRegex(
            ValueError, r"Index key '\(b'D',\)' not found in input EventSet."
        ):
            op = SelectIndexValues(input=self.node, keys="D")
            instance = SelectIndexValuesNumpyImplementation(op)
            instance.call(input=self.evset)

    def test_more_than_one_param(self):
        with self.assertRaisesRegex(
            ValueError,
            "Exactly one of keys, number or fraction must be provided.",
        ):
            SelectIndexValues(input=self.node, keys="D", number=1)

    def test_no_params(self):
        with self.assertRaisesRegex(
            ValueError,
            "Exactly one of keys, number or fraction must be provided.",
        ):
            SelectIndexValues(input=self.node)

    def test_invalid_fraction(self):
        with self.assertRaisesRegex(
            ValueError,
            "fraction must be between 0 and 1.",
        ):
            SelectIndexValues(input=self.node, fraction=-1)

        with self.assertRaisesRegex(
            ValueError,
            "fraction must be between 0 and 1.",
        ):
            SelectIndexValues(input=self.node, fraction=1.01)


if __name__ == "__main__":
    absltest.main()
