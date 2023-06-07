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

import numpy as np
import pandas as pd
from absl.testing import absltest

from temporian.core.data.dtype import DType
from temporian.core.data.node import input_node
from temporian.core.operators.binary import (
    AddOperator,
    SubtractOperator,
    MultiplyOperator,
    DivideOperator,
    FloorDivOperator,
)
from temporian.io.pandas import from_pandas
from temporian.implementation.numpy.operators.binary import (
    AddNumpyImplementation,
    SubtractNumpyImplementation,
    MultiplyNumpyImplementation,
    DivideNumpyImplementation,
    FloorDivNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class ArithmeticMultiIndexNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators,
    but using a two-level index and disordered rows."""

    def setUp(self):
        # store ids
        TRYOLABS_SHOP = 42
        GOOGLE_SHOP = 101
        # product ids
        MATE_ID = 1
        BOOK_ID = 2
        PIXEL_ID = 3

        # 2 index columns, 2 feature columns (float64 and float32)
        self.evset_1 = from_pandas(
            pd.DataFrame(
                data=[
                    [TRYOLABS_SHOP, MATE_ID, 0.0, -14.0, 1.0],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, 15.0, 2],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, 16, 3],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, 10, 4],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 0, 5],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, 8, 6],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, 3, 7],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, 4, 8],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "sales",
                    "revenue",
                ],
            ).astype(
                {"revenue": np.float32}  # Default is float64
            ),
            index_names=["store_id", "product_id"],
        )

        self.evset_2 = from_pandas(
            pd.DataFrame(
                data=[
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 3, -8.0],
                    [TRYOLABS_SHOP, MATE_ID, 0.0, 4.5, 5],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, -5.5, 3],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, -8, 2],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, 16, 1],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, 0, 2],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, 3, 4],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, 5, 3],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "costs",
                    "sales",
                ],
            ).astype(
                {"sales": np.float32}  # Default is float64
            ),
            index_names=["store_id", "product_id"],
            same_sampling_as=self.evset_1,
        )

        # Expected result after addition
        self.expected_evset_add = from_pandas(
            pd.DataFrame(
                [
                    [TRYOLABS_SHOP, MATE_ID, 0.0, -9.5, 6],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, 9.5, 5],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, 32, 4],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, 10, 6],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 3, -3],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, 0, 8],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, 6, 11],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, 9, 11],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "add_sales_costs",
                    "add_revenue_sales",
                ],
            ).astype(
                {"add_revenue_sales": np.float32}  # Default is float64
            ),
            index_names=["store_id", "product_id"],
        )
        # Expected result after subtraction
        self.expected_evset_subtract = from_pandas(
            pd.DataFrame(
                [
                    [TRYOLABS_SHOP, MATE_ID, 0.0, -18.5, -4],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, 20.5, -1],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, 0, 2],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, 10, 2],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, -3, 13],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, 16, 4],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, 0, 3],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, -1, 5],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "sub_sales_costs",
                    "sub_revenue_sales",
                ],
            ).astype(
                {"sub_revenue_sales": np.float32}  # Default is float64
            ),
            index_names=["store_id", "product_id"],
        )
        # Expected result after multiplication
        self.expected_evset_multiply = from_pandas(
            pd.DataFrame(
                [
                    [TRYOLABS_SHOP, MATE_ID, 0.0, -63, 5],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, -82.5, 6],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, 256, 3],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, 0, 8],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 0, -40],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, -64, 12],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, 9, 28],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, 20, 24],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "mult_sales_costs",
                    "mult_revenue_sales",
                ],
            ).astype(
                {"mult_revenue_sales": np.float32}  # Default is float64
            ),
            index_names=["store_id", "product_id"],
        )
        # Expected result after division
        self.expected_evset_divide = from_pandas(
            pd.DataFrame(
                [
                    [TRYOLABS_SHOP, MATE_ID, 0.0, -14 / 4.5, 0.2],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, -15 / 5.5, 2 / 3],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, 1, 3],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, np.inf, 2],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 0, -0.625],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, -1, 3],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, 1, 1.75],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, 0.8, 8 / 3],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "div_sales_costs",
                    "div_revenue_sales",
                ],
            ).astype(
                {"div_revenue_sales": np.float32}  # Default is float64
            ),
            index_names=["store_id", "product_id"],
        )

        # Expected result after floor division
        self.expected_evset_floordiv = from_pandas(
            pd.DataFrame(
                [
                    [TRYOLABS_SHOP, MATE_ID, 0.0, -4.0, 0],
                    [TRYOLABS_SHOP, MATE_ID, 1.0, -3.0, 0],
                    [TRYOLABS_SHOP, MATE_ID, 2.0, 1, 3],
                    [TRYOLABS_SHOP, BOOK_ID, 1.0, np.inf, 2],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 0, -1],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, -1, 3],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, 1, 1],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, 0, 2],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "floordiv_sales_costs",
                    "floordiv_revenue_sales",
                ],
            ).astype(
                {"floordiv_revenue_sales": np.float32}  # Default is float64
            ),
            index_names=["store_id", "product_id"],
        )

        self.node_1 = input_node(
            [("sales", DType.FLOAT64), ("revenue", DType.FLOAT32)],
            indexes=[("store_id", DType.INT64), ("product_id", DType.INT64)],
        )
        self.node_2 = input_node(
            [("costs", DType.FLOAT64), ("sales", DType.FLOAT32)],
            same_sampling_as=self.node_1,
        )

    def test_addition(self) -> None:
        operator = AddOperator(
            input_1=self.node_1,
            input_2=self.node_2,
        )

        implementation = AddNumpyImplementation(operator)
        operator.outputs["output"].check_same_sampling(self.node_1)

        operator_output = implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )["output"]
        testOperatorAndImp(self, operator, implementation)
        assertEqualEventSet(self, self.expected_evset_add, operator_output)

    def test_subtraction(self) -> None:
        operator = SubtractOperator(
            input_1=self.node_1,
            input_2=self.node_2,
        )

        implementation = SubtractNumpyImplementation(operator)
        operator_output = implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )["output"]
        testOperatorAndImp(self, operator, implementation)
        assertEqualEventSet(self, self.expected_evset_subtract, operator_output)

    def test_multiplication(self) -> None:
        operator = MultiplyOperator(
            input_1=self.node_1,
            input_2=self.node_2,
        )

        implementation = MultiplyNumpyImplementation(operator)

        operator_output = implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )["output"]
        testOperatorAndImp(self, operator, implementation)
        assertEqualEventSet(self, self.expected_evset_multiply, operator_output)

    def test_division(self) -> None:
        operator = DivideOperator(
            input_1=self.node_1,
            input_2=self.node_2,
        )

        implementation = DivideNumpyImplementation(operator)
        operator_output = implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )["output"]
        testOperatorAndImp(self, operator, implementation)
        assertEqualEventSet(self, self.expected_evset_divide, operator_output)

    def test_floordiv(self) -> None:
        operator = FloorDivOperator(
            input_1=self.node_1,
            input_2=self.node_2,
        )

        implementation = FloorDivNumpyImplementation(operator)

        operator_output = implementation.call(
            input_1=self.evset_1, input_2=self.evset_2
        )["output"]
        testOperatorAndImp(self, operator, implementation)
        assertEqualEventSet(self, self.expected_evset_floordiv, operator_output)


if __name__ == "__main__":
    absltest.main()
