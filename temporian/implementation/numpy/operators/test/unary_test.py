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
import math
import numpy as np
import pandas as pd
from absl.testing import absltest

from temporian.core.data.node import input_node
from temporian.implementation.numpy.operators.unary import (
    InvertNumpyImplementation,
    AbsNumpyImplementation,
    LogNumpyImplementation,
    IsNanNumpyImplementation,
    NotNanNumpyImplementation,
    InvertOperator,
    AbsOperator,
    LogOperator,
    IsNanOperator,
    NotNanOperator,
)
from temporian.core.data.dtypes.dtype import DType
from temporian.io.pandas import from_pandas
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
)


class UnaryNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of all unary operators"""

    def setUp(self):
        # store ids
        TRYO_SHOP = 42
        GOOGLE_SHOP = 101
        # product ids
        MATE_ID = 1
        BOOK_ID = 2
        PIXEL_ID = 3

        # 2 index columns, 2 boolean features
        self.input_evset = from_pandas(
            pd.DataFrame(
                data=[
                    [TRYO_SHOP, MATE_ID, 0.0, True, True],
                    [TRYO_SHOP, MATE_ID, 1.0, True, False],
                    [TRYO_SHOP, MATE_ID, 2.0, False, True],
                    [TRYO_SHOP, BOOK_ID, 1.0, False, False],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, False, True],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, True, False],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, False, False],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, True, True],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "boolean_1",
                    "boolean_2",
                ],
            ),
            index_names=["store_id", "product_id"],
        )

        # Expected event set after invert
        # 2 index columns, 2 boolean features
        self.expected_evset = from_pandas(
            pd.DataFrame(
                data=[
                    [TRYO_SHOP, MATE_ID, 0.0, False, False],
                    [TRYO_SHOP, MATE_ID, 1.0, False, True],
                    [TRYO_SHOP, MATE_ID, 2.0, True, False],
                    [TRYO_SHOP, BOOK_ID, 1.0, True, True],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, True, False],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, False, True],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, True, True],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, False, False],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "boolean_1",
                    "boolean_2",
                ],
            ),
            index_names=["store_id", "product_id"],
        )

        self.input_node = input_node(
            [("boolean_1", DType.BOOLEAN), ("boolean_2", DType.BOOLEAN)],
            indexes=[("store_id", DType.INT64), ("product_id", DType.INT64)],
        )

    def test_invert_boolean(self) -> None:
        """Test inversion of boolean features"""
        operator = InvertOperator(input=self.input_node)
        operator.outputs["output"].check_same_sampling(self.input_node)

        inv_implementation = InvertNumpyImplementation(operator)
        operator_output = inv_implementation.call(input=self.input_evset)
        assertEqualEventSet(
            self, self.expected_evset, operator_output["output"]
        )

    def test_invert_twice(self) -> None:
        """Negate twice to get the same input"""
        operator1 = InvertOperator(input=self.input_node)
        operator2 = InvertOperator(input=operator1.outputs["output"])
        inv1_implementation = InvertNumpyImplementation(operator1)
        inv2_implementation = InvertNumpyImplementation(operator2)
        op_out_1 = inv1_implementation.call(input=self.input_evset)["output"]
        op_out_2 = inv2_implementation.call(input=op_out_1)["output"]

        assertEqualEventSet(self, self.input_evset, op_out_2)

    def test_error_nonboolean(self) -> None:
        """Check that trying a non-boolean event raises ValueError"""
        invalid_node = input_node(
            [("boolean_1", DType.BOOLEAN), ("int_2", DType.INT32)],
            indexes=[("store_id", DType.INT32), ("product_id", DType.INT64)],
        )

        with self.assertRaisesRegex(ValueError, "bool"):
            _ = InvertOperator(input=invalid_node)

    def test_correct_abs(self) -> None:
        event_data = from_pandas(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, -2, -3]})
        )
        expected_data = from_pandas(
            pd.DataFrame({"timestamp": [1, 2, 3], "x": [1, 2, 3]})
        )
        operator = AbsOperator(input=event_data.node())
        operator_output = AbsNumpyImplementation(operator).call(
            input=event_data
        )

        assertEqualEventSet(self, expected_data, operator_output["output"])

    def test_correct_log(self) -> None:
        event_data = from_pandas(
            pd.DataFrame({"timestamp": [1, 2, 3, 4], "x": [1, np.e, 0, 10]})
        )
        expected_data = from_pandas(
            pd.DataFrame(
                {"timestamp": [1, 2, 3, 4], "x": [0, 1, -np.inf, np.log(10)]}
            )
        )
        operator = LogOperator(input=event_data.node())
        operator_output = LogNumpyImplementation(operator).call(
            input=event_data
        )

        assertEqualEventSet(self, expected_data, operator_output["output"])

    def test_correct_isnan(self) -> None:
        event_data = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3, 4, 5],
                    "x": [1, -1, np.nan, math.nan, None],
                }
            )
        )
        expected_data = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3, 4, 5],
                    "x": [False, False, True, True, True],
                }
            )
        )
        operator = IsNanOperator(input=event_data.node())
        operator_output = IsNanNumpyImplementation(operator).call(
            input=event_data
        )

        assertEqualEventSet(self, expected_data, operator_output["output"])

    def test_correct_notnan(self) -> None:
        event_data = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3, 4, 5],
                    "x": [1, -1, np.nan, math.nan, None],
                }
            )
        )
        expected_data = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3, 4, 5],
                    "x": [True, True, False, False, False],
                }
            )
        )
        operator = NotNanOperator(input=event_data.node())
        operator_output = NotNanNumpyImplementation(operator).call(
            input=event_data
        )
        assertEqualEventSet(self, expected_data, operator_output["output"])


if __name__ == "__main__":
    absltest.main()
