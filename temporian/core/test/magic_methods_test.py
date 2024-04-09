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

"""Tests magic methods on both EventSets and EventSetNodes."""

# pylint: disable=unused-argument

from typing import Union
from absl.testing import absltest
from absl.testing.parameterized import parameters
import numpy as np

from temporian.core.data.node import EventSetNode, input_node
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

EventSetNodeOrEvset = Union[EventSetNode, EventSet]

# Define parameters for all tests.
# Note that evset_X and node_X must always have matching schemas.
evset_sampling = event_set(
    timestamps=[1],
    features={"x": np.array([1.0], dtype=np.int32)},
    indexes=["x"],
    is_unix_timestamp=False,
)
node_sampling = input_node(
    features=[],
    indexes=[("x", DType.INT32)],
    is_unix_timestamp=False,
)

evset_float_1 = event_set(
    timestamps=[1],
    features={
        "x": np.array([1.0], dtype=np.int32),
        "f1": np.array([1.0], dtype=np.float32),
        "f2": np.array([2.0], dtype=np.float64),
    },
    indexes=["x"],
    same_sampling_as=evset_sampling,
)
node_float_1 = input_node(
    features=[
        ("f1", DType.FLOAT32),
        ("f2", DType.FLOAT64),
    ],
    same_sampling_as=node_sampling,
)

evset_float_2 = event_set(
    timestamps=[1],
    features={
        "x": np.array([1.0], dtype=np.int32),
        "f3": np.array([3.0], dtype=np.float32),
        "f4": np.array([4.0], dtype=np.float64),
    },
    indexes=["x"],
    same_sampling_as=evset_sampling,
)
node_float_2 = input_node(
    features=[
        ("f3", DType.FLOAT32),
        ("f4", DType.FLOAT64),
    ],
    same_sampling_as=node_sampling,
)

evset_int_1 = event_set(
    timestamps=[1],
    features={
        "x": np.array([1.0], dtype=np.int32),
        "f5": np.array([5], dtype=np.int32),
        "f6": np.array([6], dtype=np.int64),
    },
    indexes=["x"],
    same_sampling_as=evset_sampling,
)
node_int_1 = input_node(
    features=[
        ("f5", DType.INT32),
        ("f6", DType.INT64),
    ],
    same_sampling_as=node_sampling,
)

evset_int_2 = event_set(
    timestamps=[1],
    features={
        "x": np.array([1.0], dtype=np.int32),
        "f7": np.array([7], dtype=np.int32),
        "f8": np.array([8], dtype=np.int64),
    },
    indexes=["x"],
    same_sampling_as=evset_sampling,
)
node_int_2 = input_node(
    features=[
        ("f7", DType.INT32),
        ("f8", DType.INT64),
    ],
    same_sampling_as=node_sampling,
)

# Boolean types for logic operators
evset_bool_1 = evset_float_1 > evset_float_2
node_bool_1 = node_float_1 > node_float_2
evset_bool_2 = evset_int_1 > evset_int_2
node_bool_2 = node_int_1 > node_int_2


# Run all methods of this class for both EventSets and their corresponding EventSetNodes
# NOTE: all code in the test methods must work for both evsets and nodes
@parameters(
    {
        "sampling": evset_sampling,
        "float_1": evset_float_1,
        "float_2": evset_float_2,
        "int_1": evset_int_1,
        "int_2": evset_int_2,
        "bool_1": evset_bool_1,
        "bool_2": evset_bool_2,
    },
    {
        "sampling": node_sampling,
        "float_1": node_float_1,
        "float_2": node_float_2,
        "int_1": node_int_1,
        "int_2": node_int_2,
        "bool_1": node_bool_1,
        "bool_2": node_bool_2,
    },
)
class MagicMethodsTest(absltest.TestCase):
    #########################################
    ### Get/set item with square brackets ###
    #########################################
    def test_getitem_single(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = float_1["f2"]

        self.assertTrue(isinstance(out, float_1.__class__))

        out.check_same_sampling(float_1)

        self.assertEqual(len(out.schema.features), 1)
        self.assertEqual(out.schema.features[0].name, "f2")
        self.assertEqual(out.schema.features[0].dtype, DType.FLOAT64)

    def test_getitem_multiple(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = float_1[["f2", "f1"]]

        self.assertTrue(isinstance(out, float_1.__class__))

        out.check_same_sampling(float_1)

        self.assertEqual(len(out.schema.features), 2)
        self.assertEqual(out.schema.features[0].name, "f2")
        self.assertEqual(out.schema.features[0].dtype, DType.FLOAT64)
        self.assertEqual(out.schema.features[1].name, "f1")
        self.assertEqual(out.schema.features[1].dtype, DType.FLOAT32)

    def test_getitem_empty(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = float_1[[]]

        self.assertTrue(isinstance(out, float_1.__class__))

        out.check_same_sampling(float_1)
        self.assertEqual(len(out.schema.features), 0)

    def test_getitem_errors(self, float_1: EventSetNodeOrEvset, **kwargs):
        with self.assertRaises(IndexError):
            float_1["f3"]
        with self.assertRaises(IndexError):
            float_1[["f1", "f3"]]
        with self.assertRaises(TypeError):
            float_1[0]
        with self.assertRaises(TypeError):
            float_1[["f1", 0]]
        with self.assertRaises(TypeError):
            float_1[None]

    def test_setitem_fails(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        # Try to modify existent feature
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            float_1["f1"] = float_2["f3"]
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            float_1["f1"] = None

        # Try to assign inexistent feature
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            float_1["f5"] = float_2["f3"]
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            float_1["f5"] = None

        # Try to assign multiple features
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            float_1[["f1", "f2"]] = float_2[["f3", "f4"]]
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            float_1[["f1", "f2"]] = None

        # Weird cases
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            float_1[[]] = None
        with self.assertRaisesRegex(TypeError, "Cannot assign"):
            float_1[None] = None

    # ###################################
    # ### Relational binary operators ###
    # ###################################

    def _check_boolean(
        self, out: EventSetNodeOrEvset, inp: EventSetNodeOrEvset
    ):
        # Auxiliar function to check arithmetic outputs
        out.check_same_sampling(inp)
        self.assertTrue(out.schema.features[0].dtype == DType.BOOLEAN)
        self.assertTrue(out.schema.features[1].dtype == DType.BOOLEAN)

    def test_not_equal(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 != float_2
        self.assertTrue(isinstance(out.creator, NotEqualOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    def test_greater(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 > float_2
        self.assertTrue(isinstance(out.creator, GreaterOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    def test_less(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 < float_2
        self.assertTrue(isinstance(out.creator, LessOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    def test_greater_equal(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 >= float_2
        self.assertTrue(isinstance(out.creator, GreaterEqualOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    def test_less_equal(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 <= float_2
        self.assertTrue(isinstance(out.creator, LessEqualOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    # ###################################
    # ### Relational scalar operators ###
    # ###################################

    def test_not_equal_scalar(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = float_1 != 3
        self.assertTrue(isinstance(out.creator, NotEqualScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    def test_greater_scalar(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = float_1 > 3.0
        self.assertTrue(isinstance(out.creator, GreaterScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    def test_less_scalar(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = float_1 < 3
        self.assertTrue(isinstance(out.creator, LessScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    def test_greater_equal_scalar(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = float_1 >= 3.0
        self.assertTrue(isinstance(out.creator, GreaterEqualScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    def test_less_equal_scalar(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = float_1 <= 0.5
        self.assertTrue(isinstance(out.creator, LessEqualScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_boolean(out, float_1)

    # ########################
    # ### Logic operators  ###
    # ########################

    def test_logic_and(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        bool_1: EventSetNodeOrEvset,
        bool_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        # Should not work: non boolean features
        with self.assertRaises(ValueError):
            _ = float_1 & float_2

        out = bool_1 & bool_2
        self.assertTrue(isinstance(out.creator, LogicalAndOperator))
        self._check_boolean(out, bool_1)

    def test_logic_or(
        self, bool_1: EventSetNodeOrEvset, bool_2: EventSetNodeOrEvset, **kwargs
    ):
        out = bool_1 | bool_2
        self.assertTrue(isinstance(out.creator, LogicalOrOperator))
        self._check_boolean(out, bool_1)

    def test_logic_xor(
        self, bool_1: EventSetNodeOrEvset, bool_2: EventSetNodeOrEvset, **kwargs
    ):
        out = bool_1 ^ bool_2
        self.assertTrue(isinstance(out.creator, LogicalXorOperator))
        self._check_boolean(out, bool_1)

    # ###################################
    # ### Arithmetic binary operators ###
    # ###################################

    def _check_node_same_dtype(
        self, inp: EventSetNodeOrEvset, out: EventSetNodeOrEvset
    ):
        # Auxiliar function to check arithmetic outputs
        out.check_same_sampling(inp)
        self.assertTrue(
            (out.schema.features[0].dtype == inp.schema.features[0].dtype)
        )
        self.assertTrue(
            (out.schema.features[1].dtype == inp.schema.features[1].dtype)
        )

    def test_addition(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 + float_2
        self.assertTrue(isinstance(out.creator, AddOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_subtraction(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 - float_2
        self.assertTrue(isinstance(out.creator, SubtractOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_multiplication(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 * float_2
        self.assertTrue(isinstance(out.creator, MultiplyOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_division(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 / float_2
        self.assertTrue(isinstance(out.creator, DivideOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_floordiv(
        self, int_1: EventSetNodeOrEvset, int_2: EventSetNodeOrEvset, **kwargs
    ):
        # First, check that truediv is not supported for integer types
        with self.assertRaises(ValueError):
            out = int_1 / int_2

        # Check floordiv operator instead
        out = int_1 // int_2
        self.assertTrue(isinstance(out.creator, FloorDivOperator))
        self.assertTrue(out.schema.features[0].name == "f5")
        self.assertTrue(out.schema.features[1].name == "f6")
        self._check_node_same_dtype(int_1, out)

    def test_modulo(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1 % float_2
        self.assertTrue(isinstance(out.creator, ModuloOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_power(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        out = float_1**float_2
        self.assertTrue(isinstance(out.creator, PowerOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    # ###################################
    # ### Arithmetic scalar operators ###
    # ###################################

    def test_addition_scalar(
        self, float_1: EventSetNodeOrEvset, int_1: EventSetNodeOrEvset, **kwargs
    ):
        # Shouldn't work: int node and float scalar
        with self.assertRaises(ValueError):
            out = int_1 + 3.5

        # Should work: float node and int scalar
        out = float_1 + 3
        self.assertTrue(isinstance(out.creator, AddScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_right_addition_scalar(
        self, float_1: EventSetNodeOrEvset, int_1: EventSetNodeOrEvset, **kwargs
    ):
        # Shouldn't work: int node and float scalar
        with self.assertRaises(ValueError):
            out = 3.5 + int_1

        # Should work: float node and int scalar
        out = 3 + float_1
        self.assertTrue(isinstance(out.creator, AddScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_division_scalar(
        self, float_1: EventSetNodeOrEvset, int_1: EventSetNodeOrEvset, **kwargs
    ):
        # Shouldn't work: divide int node
        with self.assertRaises(ValueError):
            out = int_1 / 3

        # Should work: float node and int scalar
        out = float_1 / 3
        self.assertTrue(isinstance(out.creator, DivideScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_right_division_scalar(
        self, float_1: EventSetNodeOrEvset, int_1: EventSetNodeOrEvset, **kwargs
    ):
        # Shouldn't work: divide by int node
        with self.assertRaises(ValueError):
            out = 3 / int_1

        # Should work: divide by float node
        out = 3 / float_1
        self.assertTrue(isinstance(out.creator, DivideScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_floordiv_scalar(
        self, float_1: EventSetNodeOrEvset, int_1: EventSetNodeOrEvset, **kwargs
    ):
        # int node
        out = int_1 // 3
        self.assertTrue(out.schema.features[0].name == "f5")
        self.assertTrue(out.schema.features[1].name == "f6")

        # int node (right)
        out = 3 // int_1
        self.assertTrue(out.schema.features[0].name == "f5")
        self.assertTrue(out.schema.features[1].name == "f6")

        # float node
        out = float_1 // 3
        self.assertTrue(isinstance(out.creator, FloorDivScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_multiply_scalar(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = 3 * float_1
        self.assertTrue(isinstance(out.creator, MultiplyScalarOperator))
        out = float_1 * 3
        self.assertTrue(isinstance(out.creator, MultiplyScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_subtract_scalar(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = 3 - float_1
        self.assertTrue(isinstance(out.creator, SubtractScalarOperator))
        out = float_1 - 3
        self.assertTrue(isinstance(out.creator, SubtractScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_modulo_scalar(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = 3 % float_1
        self.assertTrue(isinstance(out.creator, ModuloScalarOperator))
        out = float_1 % 3
        self.assertTrue(isinstance(out.creator, ModuloScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    def test_power_scalar(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = 3**float_1
        self.assertTrue(isinstance(out.creator, PowerScalarOperator))
        out = float_1**3
        self.assertTrue(isinstance(out.creator, PowerScalarOperator))
        self.assertTrue(out.schema.features[0].name == "f1")
        self.assertTrue(out.schema.features[1].name == "f2")
        self._check_node_same_dtype(float_1, out)

    # ########################
    # ### Unary operators  ###
    # ########################
    def test_abs(self, float_1: EventSetNodeOrEvset, **kwargs):
        out = abs(float_1)
        self.assertTrue(isinstance(out.creator, AbsOperator))
        self._check_node_same_dtype(float_1, out)

    def test_invert(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        # Should not work: invert non-boolean types
        with self.assertRaises(ValueError):
            _ = ~float_1

        boolean_node = float_1 != float_2
        out = ~boolean_node
        self.assertTrue(isinstance(out.creator, InvertOperator))
        self._check_boolean(out, float_1)

    def test_no_truth_value(
        self,
        float_1: EventSetNodeOrEvset,
        float_2: EventSetNodeOrEvset,
        **kwargs,
    ):
        # Check that bool(node) doesn't work
        boolean_node = float_1 != float_2
        with self.assertRaisesRegex(
            ValueError,
            f"truth value of a {float_1.__class__.__name__} is ambiguous",
        ):
            bool(boolean_node)

        with self.assertRaisesRegex(
            ValueError,
            f"truth value of a {float_1.__class__.__name__} is ambiguous",
        ):
            if boolean_node:  # <- this should call bool(boolean_node)
                pass


if __name__ == "__main__":
    absltest.main()
