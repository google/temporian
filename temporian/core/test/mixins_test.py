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

"""Tests that magic methods and operators are callable in both EventSets
and Nodes."""

from absl.testing import absltest

from temporian.core.data.node import input_node
from temporian.core.data.dtype import DType
from temporian.core.operators.binary import (
    AddOperator,
    DivideOperator,
    FloorDivOperator,
    MultiplyOperator,
    SubtractOperator,
    ModuloOperator,
    PowerOperator,
)
from temporian.core.operators.binary import (
    NotEqualOperator,
    GreaterOperator,
    GreaterEqualOperator,
    LessEqualOperator,
    LessOperator,
)
from temporian.core.operators.binary import (
    LogicalAndOperator,
    LogicalOrOperator,
    LogicalXorOperator,
)
from temporian.core.operators.scalar import (
    AddScalarOperator,
    SubtractScalarOperator,
    MultiplyScalarOperator,
    DivideScalarOperator,
    FloorDivScalarOperator,
    ModuloScalarOperator,
    PowerScalarOperator,
)
from temporian.core.operators.scalar import (
    NotEqualScalarOperator,
    GreaterScalarOperator,
    GreaterEqualScalarOperator,
    LessEqualScalarOperator,
    LessScalarOperator,
)
from temporian.core.operators.unary import AbsOperator, InvertOperator
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.io import event_set


class NodeTest(absltest.TestCase):
    def setUp(self):
        self.sampling_evset = event_set(
            timestamps=[1],
            features={"x": [1.0]},
            indexes=["x"],
            is_unix_timestamp=False,
        )
        self.sampling_node = input_node(
            features=[],
            indexes=[("x", DType.INT32)],
            is_unix_timestamp=False,
        )

        # Floating point types
        self.evset_float_1 = event_set(
            timestamps=[1],
            features={
                "x": [1.0],
                "f1": [1.0],
                "f2": [2.0],
            },
            indexes=["x"],
            same_sampling_as=self.sampling_evset,
        )
        self.node_float_1 = input_node(
            features=[
                ("f1", DType.FLOAT32),
                ("f2", DType.FLOAT64),
            ],
            same_sampling_as=self.sampling_node,
        )

        self.evset_float_2 = event_set(
            timestamps=[1],
            features={
                "x": [1.0],
                "f3": [3.0],
                "f4": [4.0],
            },
            indexes=["x"],
            same_sampling_as=self.sampling_evset,
        )
        self.node_float_2 = input_node(
            features=[
                ("f3", DType.FLOAT32),
                ("f4", DType.FLOAT64),
            ],
            same_sampling_as=self.sampling_node,
        )

        # Integer types (only for division operations)
        self.evset_int_1 = event_set(
            timestamps=[1],
            features={
                "x": [1.0],
                "f5": [5],
                "f6": [6],
            },
            indexes=["x"],
            same_sampling_as=self.sampling_evset,
        )
        self.node_int_1 = input_node(
            features=[
                ("f5", DType.INT32),
                ("f6", DType.INT64),
            ],
            same_sampling_as=self.sampling_node,
        )

        self.evset_int_2 = event_set(
            timestamps=[1],
            features={
                "x": [1.0],
                "f7": [7],
                "f8": [8],
            },
            indexes=["x"],
            same_sampling_as=self.sampling_evset,
        )
        self.node_int_2 = input_node(
            features=[
                ("f7", DType.INT32),
                ("f8", DType.INT64),
            ],
            same_sampling_as=self.sampling_node,
        )

        # Boolean types for logic operators
        self.evset_bool_1 = self.evset_float_1 > self.evset_float_2
        self.node_bool_1 = self.node_float_1 > self.node_float_2
        self.evset_bool_2 = self.evset_int_1 > self.evset_int_2
        self.node_bool_2 = self.node_int_1 > self.node_int_2

    #########################################
    ### Get/set item with square brackets ###
    #########################################
    def test_node_getitem_single(self):
        node_out = self.node_float_1["f2"]
        self.assertTrue(len(node_out.schema.features) == 1)
        self.assertTrue(node_out.schema.features[0].name == "f2")
        self.assertTrue(node_out.schema.features[0].dtype == DType.FLOAT64)
        # Raises ValueError if fails
        node_out.check_same_sampling(self.node_float_1)

    # TODO: duplicate all other tests for EventSet (maybe automate it?)
    def test_evset_getitem_single(self):
        evset_out = self.evset_float_1["f2"]
        self.assertTrue(isinstance(evset_out, EventSet))
        self.assertTrue(len(evset_out.schema.features) == 1)
        self.assertTrue(evset_out.schema.features[0].name == "f2")
        self.assertTrue(evset_out.schema.features[0].dtype == DType.FLOAT64)

    def test_node_getitem_multiple(self):
        node_out = self.node_float_1[["f2"]]
        self.assertTrue(len(node_out.schema.features) == 1)
        self.assertTrue(node_out.schema.features[0].name == "f2")
        self.assertTrue(node_out.schema.features[0].dtype == DType.FLOAT64)
        node_out.check_same_sampling(self.node_float_1)

        node_out = self.node_float_1[["f2", "f1"]]
        self.assertTrue(len(node_out.schema.features) == 2)
        self.assertTrue(node_out.schema.features[0].name == "f2")
        self.assertTrue(node_out.schema.features[0].dtype == DType.FLOAT64)
        self.assertTrue(node_out.schema.features[1].name == "f1")
        self.assertTrue(node_out.schema.features[1].dtype == DType.FLOAT32)
        node_out.check_same_sampling(self.node_float_1)

        # Node with empty features
        node_out = self.node_float_1[[]]
        self.assertTrue(len(node_out.schema.features) == 0)
        node_out.check_same_sampling(self.node_float_1)

    def test_node_getitem_errors(self):
        with self.assertRaises(IndexError):
            self.node_float_1["f3"]
        with self.assertRaises(IndexError):
            self.node_float_1[["f1", "f3"]]
        with self.assertRaises(TypeError):
            self.node_float_1[0]
        with self.assertRaises(TypeError):
            self.node_float_1[["f1", 0]]
        with self.assertRaises(TypeError):
            self.node_float_1[None]

    def test_node_setitem_fails(self):
        # Try to modify existent feature
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            self.node_float_1["f1"] = self.node_float_2["f3"]
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            self.node_float_1["f1"] = None

        # Try to assign inexistent feature
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            self.node_float_1["f5"] = self.node_float_2["f3"]
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            self.node_float_1["f5"] = None

        # Try to assign multiple features
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            self.node_float_1[["f1", "f2"]] = self.node_float_2[["f3", "f4"]]
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            self.node_float_1[["f1", "f2"]] = None

        # Weird cases
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            self.node_float_1[[]] = None
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            self.node_float_1[None] = None

    ###################################
    ### Relational binary operators ###
    ###################################

    def _check_node_boolean(self, node_out):
        # Auxiliar function to check arithmetic outputs
        self.assertTrue(
            node_out.sampling_node is self.sampling_node.sampling_node
        )
        self.assertTrue(node_out.schema.features[0].dtype == DType.BOOLEAN)
        self.assertTrue(node_out.schema.features[1].dtype == DType.BOOLEAN)

    def test_node_not_equal(self):
        node_out = self.node_float_1 != self.node_float_2
        self.assertTrue(isinstance(node_out.creator, NotEqualOperator))
        self.assertTrue(node_out.schema.features[0].name == "ne_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "ne_f2_f4")
        self._check_node_boolean(node_out)

    def test_node_greater(self):
        node_out = self.node_float_1 > self.node_float_2
        self.assertTrue(isinstance(node_out.creator, GreaterOperator))
        self.assertTrue(node_out.schema.features[0].name == "gt_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "gt_f2_f4")
        self._check_node_boolean(node_out)

    def test_node_less(self):
        node_out = self.node_float_1 < self.node_float_2
        self.assertTrue(isinstance(node_out.creator, LessOperator))
        self.assertTrue(node_out.schema.features[0].name == "lt_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "lt_f2_f4")
        self._check_node_boolean(node_out)

    def test_node_greater_equal(self):
        node_out = self.node_float_1 >= self.node_float_2
        self.assertTrue(isinstance(node_out.creator, GreaterEqualOperator))
        self.assertTrue(node_out.schema.features[0].name == "ge_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "ge_f2_f4")
        self._check_node_boolean(node_out)

    def test_node_less_equal(self):
        node_out = self.node_float_1 <= self.node_float_2
        self.assertTrue(isinstance(node_out.creator, LessEqualOperator))
        self.assertTrue(node_out.schema.features[0].name == "le_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "le_f2_f4")
        self._check_node_boolean(node_out)

    ###################################
    ### Relational scalar operators ###
    ###################################

    def test_node_not_equal_scalar(self):
        node_out = self.node_float_1 != 3
        self.assertTrue(isinstance(node_out.creator, NotEqualScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_boolean(node_out)

    def test_node_greater_scalar(self):
        node_out = self.node_float_1 > 3.0
        self.assertTrue(isinstance(node_out.creator, GreaterScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_boolean(node_out)

    def test_node_less_scalar(self):
        node_out = self.node_float_1 < 3
        self.assertTrue(isinstance(node_out.creator, LessScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_boolean(node_out)

    def test_node_greater_equal_scalar(self):
        node_out = self.node_float_1 >= 3.0
        self.assertTrue(
            isinstance(node_out.creator, GreaterEqualScalarOperator)
        )
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_boolean(node_out)

    def test_node_less_equal_scalar(self):
        node_out = self.node_float_1 <= 0.5
        self.assertTrue(isinstance(node_out.creator, LessEqualScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_boolean(node_out)

    ########################
    ### Logic operators  ###
    ########################

    def test_node_logic_and(self):
        # Should not work: non boolean features
        with self.assertRaises(ValueError):
            _ = self.node_float_1 & self.node_float_2

        node_out = self.node_bool_1 & self.node_bool_2
        self.assertTrue(isinstance(node_out.creator, LogicalAndOperator))
        self._check_node_boolean(node_out)

    def test_node_logic_or(self):
        node_out = self.node_bool_1 | self.node_bool_2
        self.assertTrue(isinstance(node_out.creator, LogicalOrOperator))
        self._check_node_boolean(node_out)

    def test_node_logic_xor(self):
        node_out = self.node_bool_1 ^ self.node_bool_2
        self.assertTrue(isinstance(node_out.creator, LogicalXorOperator))
        self._check_node_boolean(node_out)

    ###################################
    ### Arithmetic binary operators ###
    ###################################

    def _check_node_same_dtype(self, node_in, node_out):
        # Auxiliar function to check arithmetic outputs
        self.assertTrue(
            node_out.sampling_node is self.sampling_node.sampling_node
        )
        self.assertTrue(
            (
                node_out.schema.features[0].dtype
                == node_in.schema.features[0].dtype
            )
        )
        self.assertTrue(
            (
                node_out.schema.features[1].dtype
                == node_in.schema.features[1].dtype
            )
        )

    def test_node_addition(self):
        node_out = self.node_float_1 + self.node_float_2
        self.assertTrue(isinstance(node_out.creator, AddOperator))
        self.assertTrue(node_out.schema.features[0].name == "add_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "add_f2_f4")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_subtraction(self):
        node_out = self.node_float_1 - self.node_float_2
        self.assertTrue(isinstance(node_out.creator, SubtractOperator))
        self.assertTrue(node_out.schema.features[0].name == "sub_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "sub_f2_f4")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_multiplication(self):
        node_out = self.node_float_1 * self.node_float_2
        self.assertTrue(isinstance(node_out.creator, MultiplyOperator))
        self.assertTrue(node_out.schema.features[0].name == "mult_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "mult_f2_f4")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_division(self):
        node_out = self.node_float_1 / self.node_float_2
        self.assertTrue(isinstance(node_out.creator, DivideOperator))
        self.assertTrue(node_out.schema.features[0].name == "div_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "div_f2_f4")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_floordiv(self):
        # First, check that truediv is not supported for integer types
        with self.assertRaises(ValueError):
            node_out = self.node_int_1 / self.node_int_2

        # Check floordiv operator instead
        node_out = self.node_int_1 // self.node_int_2
        self.assertTrue(isinstance(node_out.creator, FloorDivOperator))
        self.assertTrue(node_out.schema.features[0].name == "floordiv_f5_f7")
        self.assertTrue(node_out.schema.features[1].name == "floordiv_f6_f8")
        self._check_node_same_dtype(self.node_int_1, node_out)

    def test_node_modulo(self):
        node_out = self.node_float_1 % self.node_float_2
        self.assertTrue(isinstance(node_out.creator, ModuloOperator))
        self.assertTrue(node_out.schema.features[0].name == "mod_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "mod_f2_f4")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_power(self):
        node_out = self.node_float_1**self.node_float_2
        self.assertTrue(isinstance(node_out.creator, PowerOperator))
        self.assertTrue(node_out.schema.features[0].name == "pow_f1_f3")
        self.assertTrue(node_out.schema.features[1].name == "pow_f2_f4")
        self._check_node_same_dtype(self.node_float_1, node_out)

    ###################################
    ### Arithmetic scalar operators ###
    ###################################

    def test_node_addition_scalar(self):
        # Shouldn't work: int node and float scalar
        with self.assertRaises(ValueError):
            node_out = self.node_int_1 + 3.5

        # Should work: float node and int scalar
        node_out = self.node_float_1 + 3
        self.assertTrue(isinstance(node_out.creator, AddScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_right_addition_scalar(self):
        # Shouldn't work: int node and float scalar
        with self.assertRaises(ValueError):
            node_out = 3.5 + self.node_int_1

        # Should work: float node and int scalar
        node_out = 3 + self.node_float_1
        self.assertTrue(isinstance(node_out.creator, AddScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_division_scalar(self):
        # Shouldn't work: divide int node
        with self.assertRaises(ValueError):
            node_out = self.node_int_1 / 3

        # Should work: float node and int scalar
        node_out = self.node_float_1 / 3
        self.assertTrue(isinstance(node_out.creator, DivideScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_right_division_scalar(self):
        # Shouldn't work: divide by int node
        with self.assertRaises(ValueError):
            node_out = 3 / self.node_int_1

        # Should work: divide by float node
        node_out = 3 / self.node_float_1
        self.assertTrue(isinstance(node_out.creator, DivideScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_floordiv_scalar(self):
        # int node
        node_out = self.node_int_1 // 3
        self.assertTrue(node_out.schema.features[0].name == "f5")
        self.assertTrue(node_out.schema.features[1].name == "f6")

        # int node (right)
        node_out = 3 // self.node_int_1
        self.assertTrue(node_out.schema.features[0].name == "f5")
        self.assertTrue(node_out.schema.features[1].name == "f6")

        # float node
        node_out = self.node_float_1 // 3
        self.assertTrue(isinstance(node_out.creator, FloorDivScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_multiply_scalar(self):
        node_out = 3 * self.node_float_1
        self.assertTrue(isinstance(node_out.creator, MultiplyScalarOperator))
        node_out = self.node_float_1 * 3
        self.assertTrue(isinstance(node_out.creator, MultiplyScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_subtract_scalar(self):
        node_out = 3 - self.node_float_1
        self.assertTrue(isinstance(node_out.creator, SubtractScalarOperator))
        node_out = self.node_float_1 - 3
        self.assertTrue(isinstance(node_out.creator, SubtractScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_modulo_scalar(self):
        node_out = 3 % self.node_float_1
        self.assertTrue(isinstance(node_out.creator, ModuloScalarOperator))
        node_out = self.node_float_1 % 3
        self.assertTrue(isinstance(node_out.creator, ModuloScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_power_scalar(self):
        node_out = 3**self.node_float_1
        self.assertTrue(isinstance(node_out.creator, PowerScalarOperator))
        node_out = self.node_float_1**3
        self.assertTrue(isinstance(node_out.creator, PowerScalarOperator))
        self.assertTrue(node_out.schema.features[0].name == "f1")
        self.assertTrue(node_out.schema.features[1].name == "f2")
        self._check_node_same_dtype(self.node_float_1, node_out)

    ########################
    ### Unary operators  ###
    ########################
    def test_node_abs(self):
        node_out = abs(self.node_float_1)
        self.assertTrue(isinstance(node_out.creator, AbsOperator))
        self._check_node_same_dtype(self.node_float_1, node_out)

    def test_node_invert(self):
        # Should not work: invert non-boolean types
        with self.assertRaises(ValueError):
            _ = ~self.node_float_1

        boolean_node = self.node_float_1 != self.node_float_2
        node_out = ~boolean_node
        self.assertTrue(isinstance(node_out.creator, InvertOperator))
        self._check_node_boolean(node_out)

    def test_node_no_truth_value(self):
        # Check that bool(node) doesn't work
        boolean_node = self.node_float_1 != self.node_float_2
        with self.assertRaisesRegex(
            ValueError, "truth value of a Node is ambiguous"
        ):
            bool(boolean_node)

        with self.assertRaisesRegex(
            ValueError, "truth value of a Node is ambiguous"
        ):
            if boolean_node:  # <- this should call bool(boolean_node)
                pass


if __name__ == "__main__":
    absltest.main()
