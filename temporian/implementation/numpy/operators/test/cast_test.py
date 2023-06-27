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

from temporian.core.data.node import input_node
from temporian.implementation.numpy.operators.cast import (
    CastNumpyImplementation,
)
from temporian.core.operators.cast import CastOperator, cast
from temporian.core.data.dtype import DType
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.io.pandas import from_pandas
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)
from temporian.core.evaluation import evaluate


class CastNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators,
    but using a two-level index and disordered rows."""

    def setUp(self):
        # store ids
        TRYO_SHOP = 42
        GOOGLE_SHOP = 101
        # product ids
        MATE_ID = 1
        BOOK_ID = 2
        PIXEL_ID = 3

        # Some numbers above int32 and below int64
        abv_i32 = np.iinfo(np.int32).max + 1
        blw_i64 = np.finfo(np.float64).min  # below float32 too

        # 2 index columns, 3 features: float64, int64, str, boolean
        self.input_evset = from_pandas(
            pd.DataFrame(
                data=[
                    [TRYO_SHOP, MATE_ID, 0.0, -14.0, abv_i32, "1.2", True],
                    [TRYO_SHOP, MATE_ID, 1.0, 15.0, 0, "2.0", True],
                    [TRYO_SHOP, MATE_ID, 2.0, 16, 3, "000000", True],
                    [TRYO_SHOP, BOOK_ID, 1.0, 10, 4, "100", False],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 0, 5, "3.2", False],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, 0, 6, "001", True],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, -1, -8, "-3.5", False],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, blw_i64, 8, "-2", False],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    # The following are just tricky column names :)
                    "f_float_64",
                    "f_int_64",
                    "f_str",
                    "f_boolean",
                ],
            ),
            indexes=["store_id", "product_id"],
        )

        self.input_node = self.input_evset.node()

        # Expected EventSet when applying some downcast operations
        self.expected_evset_1 = from_pandas(
            pd.DataFrame(
                data=[
                    # Note: astype() below will truncate above/below numbers
                    [TRYO_SHOP, MATE_ID, 0.0, -14.0, abv_i32, 1.2, 1],
                    [TRYO_SHOP, MATE_ID, 1.0, 15.0, 0, 2, 1],
                    [TRYO_SHOP, MATE_ID, 2.0, 16, 3, 0, 1],
                    [TRYO_SHOP, BOOK_ID, 1.0, 10, 4, 100, 0],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 0, 5, 3.2, 0],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, 0, 6, 1, 1],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, -1, -8, -3.5, 0],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, blw_i64, 8, -2, 0],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "f_float_64",
                    "f_int_64",
                    "f_str",
                    "f_boolean",
                ],
                # Even more tricky: these columns won't match their type
            ).astype({"f_float_64": np.float32, "f_int_64": np.int32}),
            indexes=["store_id", "product_id"],
        )

        # Expected when converting everything to float32
        self.expected_evset_2 = from_pandas(
            pd.DataFrame(
                data=[
                    # Note: astype() below will truncate above/below numbers
                    [TRYO_SHOP, MATE_ID, 0.0, -14.0, abv_i32, 1.2, 1],
                    [TRYO_SHOP, MATE_ID, 1.0, 15.0, 0, 2, 1],
                    [TRYO_SHOP, MATE_ID, 2.0, 16, 3, 0, 1],
                    [TRYO_SHOP, BOOK_ID, 1.0, 10, 4, 100, 0],
                    [GOOGLE_SHOP, BOOK_ID, 1.0, 0, 5, 3.2, 0],
                    [GOOGLE_SHOP, BOOK_ID, 2.0, 0, 6, 1, 1],
                    [GOOGLE_SHOP, PIXEL_ID, 0.0, -1, -8, -3.5, 0],
                    [GOOGLE_SHOP, PIXEL_ID, 1.0, blw_i64, 8, -2, 0],
                ],
                columns=[
                    "store_id",
                    "product_id",
                    "timestamp",
                    "f_float_64",
                    "f_int_64",
                    "f_str",
                    "f_boolean",
                ],
            ).astype(
                {
                    "f_float_64": np.float32,
                    "f_int_64": np.float32,
                    "f_str": np.float32,
                    "f_boolean": np.float32,
                }
            ),
            indexes=["store_id", "product_id"],
        )

    def test_cast_manual(self) -> None:
        node = input_node([("x", DType.FLOAT32), ("y", DType.FLOAT32)])
        op = CastOperator(node, check_overflow=True, dtype=DType.INT64)
        imp = CastNumpyImplementation(op)
        testOperatorAndImp(self, op, imp)

    def test_cast_op_by_feature(self) -> None:
        """Test correct casting by feat. names, without overflow check."""

        output_node = cast(
            self.input_node,
            target={
                "f_float_64": DType.FLOAT32,
                "f_int_64": DType.INT32,
                "f_str": DType.FLOAT64,
                "f_boolean": DType.INT64,
            },
            check_overflow=False,
        )
        output_node.check_same_sampling(self.input_node)

        output_evset = run(
            output_node,
            {self.input_node: self.input_evset},
            check_execution=True,
        )
        assert isinstance(output_evset, EventSet)
        assertEqualEventSet(self, output_evset, self.expected_evset_1)

    def test_cast_op_by_dtype(self) -> None:
        """Test correct casting by origin DType, without overflow check."""

        output_node = cast(
            self.input_node,
            target={
                DType.FLOAT64: DType.FLOAT32,
                DType.INT64: DType.INT32,
                DType.STRING: DType.FLOAT64,
                DType.BOOLEAN: DType.INT64,
                DType.INT32: DType.INT64,  # This one should have no effect
            },
            check_overflow=False,
        )
        output_evset = run(
            output_node,
            {self.input_node: self.input_evset},
            check_execution=True,
        )
        assert isinstance(output_evset, EventSet)
        assertEqualEventSet(self, output_evset, self.expected_evset_1)

    def test_cast_to_dtype(self) -> None:
        """Test correct casting everything to float32, no overflow check."""

        output_node = cast(
            self.input_node,
            target=DType.FLOAT32,
            check_overflow=False,
        )
        output_evset = run(
            output_node,
            {self.input_node: self.input_evset},
            check_execution=True,
        )
        assert isinstance(output_evset, EventSet)
        assertEqualEventSet(self, output_evset, self.expected_evset_2)

    def test_cast_no_effect(self) -> None:
        """Test the case in which there's nothing to do actually."""

        output_node = cast(
            self.input_node,
            target={
                DType.FLOAT64: DType.FLOAT64,
                DType.INT64: DType.INT64,
                DType.STRING: DType.STRING,
                DType.BOOLEAN: DType.BOOLEAN,
            },
            check_overflow=False,
        )
        self.assertTrue(output_node is self.input_node)

    def test_overflow_int64_int32(self) -> None:
        """Test overflow check for int32, coming from int64."""

        output_node = cast(
            self.input_node,
            target={DType.INT64: DType.INT32},
            check_overflow=True,
        )

        with self.assertRaisesRegex(ValueError, "Overflow"):
            _ = run(
                output_node,
                {self.input_node: self.input_evset},
                check_execution=True,
            )

    def test_overflow_float64_float32(self) -> None:
        """Test overflow check for float32, coming from float64."""

        output_node = cast(
            self.input_node,
            target={DType.FLOAT64: DType.FLOAT32},
            check_overflow=True,
        )

        with self.assertRaisesRegex(ValueError, "Overflow"):
            _ = run(
                output_node,
                {self.input_node: self.input_evset},
                check_execution=True,
            )

    def test_no_overflow_boolean(self) -> None:
        """Test that no overflow error is raised when
        converting to boolean type"""

        output_node = cast(
            self.input_node,
            target={"f_int_64": DType.BOOLEAN, "f_float_64": DType.BOOLEAN},
            check_overflow=True,
        )

        _ = run(
            output_node,
            {self.input_node: self.input_evset},
            check_execution=True,
        )

    def test_python_types(self):
        input_data = event_set(timestamps=[1, 2], features={"a": [1, 2]})

        # All features
        output_node = cast(input_data.node(), float)
        self.assertEqual(output_node.features[0].dtype, DType.FLOAT64)

        # Map type->type
        output_node = cast(input_data.node(), {int: float})
        self.assertEqual(output_node.features[0].dtype, DType.FLOAT64)

        # Map feature->type
        output_node = cast(input_data.node(), {"a": float})
        self.assertEqual(output_node.features[0].dtype, DType.FLOAT64)


if __name__ == "__main__":
    absltest.main()
