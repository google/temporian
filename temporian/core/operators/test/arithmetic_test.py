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
from temporian.core.data.dtype import DType
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.binary import (
    AddOperator,
    DivideOperator,
    FloorDivOperator,
    MultiplyOperator,
    SubtractOperator,
)


class ArithmeticOperatorsTest(absltest.TestCase):
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

    def test_addition(self):
        node_out = self.node_1 + self.node_2
        print(f"Creator: type={type(node_out.creator)} {node_out.creator}")
        assert isinstance(node_out.creator, AddOperator)
        assert node_out.sampling is self.sampling
        assert node_out.features[0].creator is node_out.creator
        assert node_out.features[1].creator is node_out.creator
        assert node_out.features[0].name == "add_f1_f3"
        assert node_out.features[1].name == "add_f2_f4"
        assert node_out.features[0].dtype == DType.FLOAT32
        assert node_out.features[1].dtype == DType.FLOAT64

    def test_subtraction(self):
        node_out = self.node_1 - self.node_2
        assert isinstance(node_out.creator, SubtractOperator)
        assert node_out.sampling is self.sampling
        assert node_out.features[0].creator is node_out.creator
        assert node_out.features[1].creator is node_out.creator
        assert node_out.features[0].name == "sub_f1_f3"
        assert node_out.features[1].name == "sub_f2_f4"
        assert node_out.features[0].dtype == DType.FLOAT32
        assert node_out.features[1].dtype == DType.FLOAT64

    def test_multiplication(self):
        node_out = self.node_1 * self.node_2
        assert isinstance(node_out.creator, MultiplyOperator)
        assert node_out.sampling is self.sampling
        assert node_out.features[0].creator is node_out.creator
        assert node_out.features[1].creator is node_out.creator
        assert node_out.features[0].name == "mult_f1_f3"
        assert node_out.features[1].name == "mult_f2_f4"
        assert node_out.features[0].dtype == DType.FLOAT32
        assert node_out.features[1].dtype == DType.FLOAT64

    def test_division(self):
        node_out = self.node_1 / self.node_2
        assert isinstance(node_out.creator, DivideOperator)
        assert node_out.sampling is self.sampling
        assert node_out.features[0].creator is node_out.creator
        assert node_out.features[1].creator is node_out.creator
        assert node_out.features[0].name == "div_f1_f3"
        assert node_out.features[1].name == "div_f2_f4"
        assert node_out.features[0].dtype == DType.FLOAT32
        assert node_out.features[1].dtype == DType.FLOAT64

    def test_floordiv(self):
        # First, check that truediv is not supported for integer types
        with self.assertRaisesRegex(
            ValueError, "Cannot use the divide operator"
        ):
            node_out = self.node_3 / self.node_4

        # Check floordiv operator instead
        node_out = self.node_3 // self.node_4
        assert isinstance(node_out.creator, FloorDivOperator)
        assert node_out.sampling is self.sampling
        assert node_out.features[0].creator is node_out.creator
        assert node_out.features[1].creator is node_out.creator
        assert node_out.features[0].name == "floordiv_f5_f7"
        assert node_out.features[1].name == "floordiv_f6_f8"
        assert node_out.features[0].dtype == DType.INT32
        assert node_out.features[1].dtype == DType.INT64


if __name__ == "__main__":
    absltest.main()
