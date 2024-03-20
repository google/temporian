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
from temporian.beam.test.utils import check_beam_implementation
from temporian.core.operators.cast import cast
from temporian.test.utils import f64, i64

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
        input_data = self.evset
        if feats is not None:
            input_data = input_data[feats]

        output_node = cast(
            input_data.node(),
            cast_arg,
            check_overflow=False,
        )

        check_beam_implementation(
            self, input_data=input_data, output_node=output_node
        )


if __name__ == "__main__":
    absltest.main()
