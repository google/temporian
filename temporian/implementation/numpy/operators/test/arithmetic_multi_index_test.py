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

from temporian.core.data.event import Event, Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.arithmetic import (
    AdditionOperator,
    SubtractionOperator,
    MultiplicationOperator,
    DivisionOperator,
)
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.arithmetic import (
    AdditionNumpyImplementation,
    SubtractionNumpyImplementation,
    MultiplicationNumpyImplementation,
    DivisionNumpyImplementation,
)
from temporian.core.data import dtype as dtype_lib


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
        self.numpy_event_1 = NumpyEvent.from_dataframe(
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

        self.numpy_event_2 = NumpyEvent.from_dataframe(
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
        )

        # Expected result after addition PER_FEATURE_IDX
        self.numpy_expected_add = NumpyEvent.from_dataframe(
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
        # Expected result after subtraction PER_FEATURE_IDX
        self.numpy_expected_subtract = NumpyEvent.from_dataframe(
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
        # Expected result after multiplication PER_FEATURE_IDX
        self.numpy_expected_multiply = NumpyEvent.from_dataframe(
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
        # Expected result after division PER_FEATURE_IDX
        self.numpy_expected_divide = NumpyEvent.from_dataframe(
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

        self.numpy_event_2.sampling = self.numpy_event_1.sampling

        self.sampling = Sampling(["store_id", "product_id"])
        self.event_1 = Event(
            [
                Feature("sales", dtype_lib.FLOAT64),
                Feature("revenue", dtype_lib.FLOAT32),
            ],
            sampling=self.sampling,
            creator=None,
        )
        self.event_2 = Event(
            [
                Feature("costs", dtype_lib.FLOAT64),
                Feature("sales", dtype_lib.FLOAT32),
            ],
            sampling=self.sampling,
            creator=None,
        )

    def test_correct_addition(self) -> None:
        """Test correct addition operator."""

        operator = AdditionOperator(
            event_1=self.event_1,
            event_2=self.event_2,
        )

        add_implementation = AdditionNumpyImplementation(operator)

        operator_output = add_implementation.call(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertTrue(self.numpy_expected_add == operator_output["event"])

    def test_correct_subtraction(self) -> None:
        """Test correct subtraction operator."""

        operator = SubtractionOperator(
            event_1=self.event_1,
            event_2=self.event_2,
        )

        sub_implementation = SubtractionNumpyImplementation(operator)
        operator_output = sub_implementation.call(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )
        self.assertTrue(
            self.numpy_expected_subtract == operator_output["event"]
        )

    def test_correct_multiplication(self) -> None:
        """Test correct multiplication operator."""

        operator = MultiplicationOperator(
            event_1=self.event_1,
            event_2=self.event_2,
        )

        mult_implementation = MultiplicationNumpyImplementation(operator)

        operator_output = mult_implementation.call(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertTrue(
            self.numpy_expected_multiply == operator_output["event"]
        )

    def test_correct_division(self) -> None:
        """Test correct division operator."""

        operator = DivisionOperator(
            event_1=self.event_1,
            event_2=self.event_2,
        )

        div_implementation = DivisionNumpyImplementation(operator)

        operator_output = div_implementation.call(
            event_1=self.numpy_event_1, event_2=self.numpy_event_2
        )

        self.assertTrue(self.numpy_expected_divide == operator_output["event"])


if __name__ == "__main__":
    absltest.main()