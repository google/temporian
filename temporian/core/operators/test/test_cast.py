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
from absl.testing import absltest, parameterized

from temporian.core.data.dtype import DType
from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, f32, f64, i32, i64


# Some numbers above int32 and below int64
ABOVE_i32 = np.iinfo(np.int32).max + 1
BELOW_i64 = np.finfo(np.float64).min  # below float32 too


class CastTest(parameterized.TestCase):
    def setUp(self):
        self.evset = event_set(
            timestamps=[0, 1, 1, 2],
            features={
                "idx": [1, 1, 2, 2],
                "f64": f64([-14.0, 0.0, BELOW_i64, 10.0]),
                "i64": i64([ABOVE_i32, 0, -1, 10]),
                "str": ["1.2", "000000", "-3.5", "001"],
                "bool": [True, True, False, False],
            },
            indexes=["idx"],
        )

    @parameterized.parameters(
        {
            # Cast by feature name
            "cast_arg": {
                "f64": DType.FLOAT32,
                "i64": DType.INT32,
                "str": DType.FLOAT32,
                "bool": DType.INT32,
            }
        },
        {
            # Cast by python dtypes
            "cast_arg": {
                float: DType.FLOAT32,
                int: DType.INT32,
                str: DType.FLOAT32,
                bool: DType.INT32,
            }
        },
        {
            # Cast by temporian dtypes
            "cast_arg": {
                DType.FLOAT64: DType.FLOAT32,
                DType.INT64: DType.INT32,
                DType.STRING: DType.FLOAT32,
                DType.BOOLEAN: DType.INT32,
            }
        },
        {
            # Cast float64 and str to float32
            "cast_arg": DType.FLOAT32,
            "feats": ["f64", "str"],
        },
    )
    def test_half_precision(self, cast_arg, feats=None) -> None:
        all_expected = event_set(
            timestamps=[0, 1, 1, 2],
            features={
                "idx": [1, 1, 2, 2],
                "f64": f32([-14.0, 0.0, BELOW_i64, 10.0]),
                "i64": i32([ABOVE_i32, 0, -1, 10]),
                "str": f32([1.2, 000000, -3.5, 1]),
                "bool": i32([1, 1, 0, 0]),
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )

        # Use only some feats
        if feats is None:
            feats = self.evset.schema.feature_names()
        evset = self.evset[feats]
        expected = all_expected[feats]

        result = evset.cast(
            cast_arg,
            check_overflow=False,
        )
        assertOperatorResult(self, result, expected)

    def test_cast_no_effect(self) -> None:
        """Test the case in which there's nothing to do actually."""
        result = self.evset.cast(
            target={
                "f64": float,
                "i64": int,
                "str": str,
                "bool": bool,
            },
            check_overflow=False,
        )
        self.assertTrue(self.evset.node() is result.node())

    def test_overflow_int64_int32(self) -> None:
        """Test overflow check for int32, coming from int64."""
        with self.assertRaisesRegex(ValueError, "Overflow"):
            self.evset.cast(
                target={DType.INT64: DType.INT32},
                check_overflow=True,
            )

    def test_overflow_float64_float32(self) -> None:
        """Test overflow check for float32, coming from float64."""
        with self.assertRaisesRegex(ValueError, "Overflow"):
            self.evset.cast(
                target={DType.FLOAT64: DType.FLOAT32},
                check_overflow=True,
            )

    def test_no_overflow_boolean(self) -> None:
        """Test that no overflow error is raised when
        converting to boolean type"""
        expected = event_set(
            timestamps=[0, 1, 1, 2],
            features={
                "idx": [1, 1, 2, 2],
                "f64": [True, False, True, True],
                "i64": [True, False, True, True],
            },
            indexes=["idx"],
            same_sampling_as=self.evset,
        )
        result = self.evset[["f64", "i64"]].cast(bool, check_overflow=True)
        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
