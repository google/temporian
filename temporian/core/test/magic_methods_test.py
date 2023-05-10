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

from temporian.core.data import node as node_lib
from temporian.core.test import utils
from temporian.core.data.dtype import DType
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
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


class MagicMethodsTest(absltest.TestCase):
    def setUp(self):
        self.sampling = Sampling(
            index_levels=[("x", DType.INT32)], is_unix_timestamp=False
        )

        # Events with floating point types
        self.node_1 = node_lib.input_node(
            features=[
                Feature("f1", DType.FLOAT32),
                Feature("f2", DType.FLOAT64),
            ],
            sampling=self.sampling,
        )
        self.node_2 = node_lib.input_node(
            features=[
                Feature("f3", DType.FLOAT32),
                Feature("f4", DType.FLOAT64),
            ],
            sampling=self.sampling,
        )

        # Events with integer types (only for division operations)
        self.node_3 = node_lib.input_node(
            features=[
                Feature("f5", DType.INT32),
                Feature("f6", DType.INT64),
            ],
            sampling=self.sampling,
        )
        self.node_4 = node_lib.input_node(
            features=[
                Feature("f7", DType.INT32),
                Feature("f8", DType.INT64),
            ],
            sampling=self.sampling,
        )

    def test_hash_map(self):
        """
        Test that the Node can be used as dict key
        This is the reason to not overwrite __eq__ in Node
        """
        node_list = []
        node_map = {}
        for i in range(100):
            node_name = f"node_{i}"
            node = utils.create_input_node(name=node_name)
            node_list.append(node)
            node_map[node] = node_name

        for idx, node in enumerate(node_list):
            assert node_map[node] == node.name
            assert idx == node_list.index(node)
            assert node in node_list
            assert node in node_map

    ###################################
    ### Relational binary operators ###
    ###################################

    def _check_node_boolean(self, node_out):
        # Auxiliar function to check arithmetic outputs
        assert node_out.sampling is self.sampling
        assert node_out.features[0].creator is node_out.creator
        assert node_out.features[1].creator is node_out.creator
        assert node_out.features[0].dtype == DType.BOOLEAN
        assert node_out.features[1].dtype == DType.BOOLEAN

    def test_not_equal(self):
        node_out = self.node_1 != self.node_2
        assert isinstance(node_out.creator, NotEqualOperator)
        assert node_out.features[0].name == "ne_f1_f3"
        assert node_out.features[1].name == "ne_f2_f4"
        self._check_node_boolean(node_out)

    def test_greater(self):
        node_out = self.node_1 > self.node_2
        assert isinstance(node_out.creator, GreaterOperator)
        assert node_out.features[0].name == "gt_f1_f3"
        assert node_out.features[1].name == "gt_f2_f4"
        self._check_node_boolean(node_out)

    def test_less(self):
        node_out = self.node_1 < self.node_2
        assert isinstance(node_out.creator, LessOperator)
        assert node_out.features[0].name == "lt_f1_f3"
        assert node_out.features[1].name == "lt_f2_f4"
        self._check_node_boolean(node_out)

    def test_greater_equal(self):
        node_out = self.node_1 >= self.node_2
        assert isinstance(node_out.creator, GreaterEqualOperator)
        assert node_out.features[0].name == "ge_f1_f3"
        assert node_out.features[1].name == "ge_f2_f4"
        self._check_node_boolean(node_out)

    def test_less_equal(self):
        node_out = self.node_1 <= self.node_2
        assert isinstance(node_out.creator, LessEqualOperator)
        assert node_out.features[0].name == "le_f1_f3"
        assert node_out.features[1].name == "le_f2_f4"
        self._check_node_boolean(node_out)

    ###################################
    ### Relational scalar operators ###
    ###################################

    def test_not_equal_scalar(self):
        node_out = self.node_1 != 3
        assert isinstance(node_out.creator, NotEqualScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_boolean(node_out)

    def test_greater_scalar(self):
        node_out = self.node_1 > 3.0
        assert isinstance(node_out.creator, GreaterScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_boolean(node_out)

    def test_less_scalar(self):
        node_out = self.node_1 < 3
        assert isinstance(node_out.creator, LessScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_boolean(node_out)

    def test_greater_equal_scalar(self):
        node_out = self.node_1 >= 3.0
        assert isinstance(node_out.creator, GreaterEqualScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_boolean(node_out)

    def test_less_equal_scalar(self):
        node_out = self.node_1 <= 0.5
        assert isinstance(node_out.creator, LessEqualScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_boolean(node_out)

    ########################
    ### Logic operators  ###
    ########################

    def test_logic_and(self):
        # Should not work: non boolean features
        with self.assertRaises(ValueError):
            _ = self.node_1 & self.node_2

        node_out = (self.node_1 > self.node_2) & (self.node_3 < self.node_4)
        assert isinstance(node_out.creator, LogicalAndOperator)
        self._check_node_boolean(node_out)

    def test_logic_or(self):
        node_out = (self.node_1 > self.node_2) | (self.node_3 < self.node_4)
        assert isinstance(node_out.creator, LogicalOrOperator)
        self._check_node_boolean(node_out)

    def test_logic_xor(self):
        node_out = (self.node_1 > self.node_2) ^ (self.node_3 < self.node_4)
        assert isinstance(node_out.creator, LogicalXorOperator)
        self._check_node_boolean(node_out)

    ###################################
    ### Arithmetic binary operators ###
    ###################################

    def _check_node_same_dtype(self, node_in, node_out):
        # Auxiliar function to check arithmetic outputs
        assert node_out.sampling is self.sampling
        assert node_out.features[0].creator is node_out.creator
        assert node_out.features[1].creator is node_out.creator
        assert node_out.features[0].dtype == node_in.features[0].dtype
        assert node_out.features[1].dtype == node_in.features[1].dtype

    def test_addition(self):
        node_out = self.node_1 + self.node_2
        assert isinstance(node_out.creator, AddOperator)
        assert node_out.features[0].name == "add_f1_f3"
        assert node_out.features[1].name == "add_f2_f4"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_subtraction(self):
        node_out = self.node_1 - self.node_2
        assert isinstance(node_out.creator, SubtractOperator)
        assert node_out.features[0].name == "sub_f1_f3"
        assert node_out.features[1].name == "sub_f2_f4"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_multiplication(self):
        node_out = self.node_1 * self.node_2
        assert isinstance(node_out.creator, MultiplyOperator)
        assert node_out.features[0].name == "mult_f1_f3"
        assert node_out.features[1].name == "mult_f2_f4"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_division(self):
        node_out = self.node_1 / self.node_2
        assert isinstance(node_out.creator, DivideOperator)
        assert node_out.features[0].name == "div_f1_f3"
        assert node_out.features[1].name == "div_f2_f4"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_floordiv(self):
        # First, check that truediv is not supported for integer types
        with self.assertRaises(ValueError):
            node_out = self.node_3 / self.node_4

        # Check floordiv operator instead
        node_out = self.node_3 // self.node_4
        assert isinstance(node_out.creator, FloorDivOperator)
        assert node_out.features[0].name == "floordiv_f5_f7"
        assert node_out.features[1].name == "floordiv_f6_f8"
        self._check_node_same_dtype(self.node_3, node_out)

    def test_modulo(self):
        node_out = self.node_1 % self.node_2
        assert isinstance(node_out.creator, ModuloOperator)
        assert node_out.features[0].name == "mod_f1_f3"
        assert node_out.features[1].name == "mod_f2_f4"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_power(self):
        node_out = self.node_1**self.node_2
        assert isinstance(node_out.creator, PowerOperator)
        assert node_out.features[0].name == "pow_f1_f3"
        assert node_out.features[1].name == "pow_f2_f4"
        self._check_node_same_dtype(self.node_1, node_out)

    ###################################
    ### Arithmetic scalar operators ###
    ###################################

    def test_addition_scalar(self):
        # Shouldn't work: int node and float scalar
        with self.assertRaises(ValueError):
            node_out = self.node_3 + 3.5

        # Should work: float node and int scalar
        node_out = self.node_1 + 3
        assert isinstance(node_out.creator, AddScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_right_addition_scalar(self):
        # Shouldn't work: int node and float scalar
        with self.assertRaises(ValueError):
            node_out = 3.5 + self.node_3

        # Should work: float node and int scalar
        node_out = 3 + self.node_1
        assert isinstance(node_out.creator, AddScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_division_scalar(self):
        # Shouldn't work: divide int node
        with self.assertRaises(ValueError):
            node_out = self.node_3 / 3

        # Should work: float node and int scalar
        node_out = self.node_1 / 3
        assert isinstance(node_out.creator, DivideScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_right_division_scalar(self):
        # Shouldn't work: divide by int node
        with self.assertRaises(ValueError):
            node_out = 3 / self.node_3

        # Should work: divide by float node
        node_out = 3 / self.node_1
        assert isinstance(node_out.creator, DivideScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_floordiv_scalar(self):
        # int node
        node_out = self.node_3 // 3
        assert node_out.features[0].name == "f5"
        assert node_out.features[1].name == "f6"

        # int node (right)
        node_out = 3 // self.node_3
        assert node_out.features[0].name == "f5"
        assert node_out.features[1].name == "f6"

        # float node
        node_out = self.node_1 // 3
        assert isinstance(node_out.creator, FloorDivScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_multiply_scalar(self):
        node_out = 3 * self.node_1
        assert isinstance(node_out.creator, MultiplyScalarOperator)
        node_out = self.node_1 * 3
        assert isinstance(node_out.creator, MultiplyScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_subtract_scalar(self):
        node_out = 3 - self.node_1
        assert isinstance(node_out.creator, SubtractScalarOperator)
        node_out = self.node_1 - 3
        assert isinstance(node_out.creator, SubtractScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_modulo_scalar(self):
        node_out = 3 % self.node_1
        assert isinstance(node_out.creator, ModuloScalarOperator)
        node_out = self.node_1 % 3
        assert isinstance(node_out.creator, ModuloScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_same_dtype(self.node_1, node_out)

    def test_power_scalar(self):
        node_out = 3**self.node_1
        assert isinstance(node_out.creator, PowerScalarOperator)
        node_out = self.node_1**3
        assert isinstance(node_out.creator, PowerScalarOperator)
        assert node_out.features[0].name == "f1"
        assert node_out.features[1].name == "f2"
        self._check_node_same_dtype(self.node_1, node_out)

    ########################
    ### Unary operators  ###
    ########################
    def test_abs(self):
        node_out = abs(self.node_1)
        assert isinstance(node_out.creator, AbsOperator)
        self._check_node_same_dtype(self.node_1, node_out)

    def test_invert(self):
        # Should not work: invert non-boolean types
        with self.assertRaises(ValueError):
            _ = ~self.node_1

        boolean_node = self.node_1 != self.node_2
        node_out = ~boolean_node
        assert isinstance(node_out.creator, InvertOperator)
        self._check_node_boolean(node_out)

    def test_no_truth_value(self):
        # Check that bool(node) doesn't work
        boolean_node = self.node_1 != self.node_2
        with self.assertRaisesRegex(
            ValueError, "truth value of a node is ambiguous"
        ):
            bool(boolean_node)

        with self.assertRaisesRegex(
            ValueError, "truth value of a node is ambiguous"
        ):
            if boolean_node:  # <- this should call bool(boolean_node)
                pass


if __name__ == "__main__":
    absltest.main()
