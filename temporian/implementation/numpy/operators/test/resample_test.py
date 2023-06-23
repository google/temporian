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
from absl.testing import absltest

import pandas as pd

from temporian.core.operators.resample import Resample
from temporian.implementation.numpy.operators.resample import (
    ResampleNumpyImplementation,
)
from temporian.io.pandas import from_pandas


class ResampleOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        evset = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [1, 5, 8, 9, 1, 1],
                    "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "b": [5, 6, 7, 8, 9, 10],
                    "c": ["A", "B", "C", "D", "E", "F"],
                    "x": [1, 1, 1, 1, 2, 2],
                }
            ),
            indexes=["x"],
        )

        sampling_evset = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [-1, 1, 6, 10, 2, 2, 1],
                    "x": [1, 1, 1, 1, 2, 2, 3],
                }
            ),
            indexes=["x"],
        )

        expected_output = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [-1, 1, 6, 10, 2, 2, 1],
                    "a": [math.nan, 1.0, 2.0, 4.0, 6.0, 6.0, math.nan],
                    "b": [0, 5, 6, 8, 10, 10, 0],
                    "c": ["", "A", "B", "D", "F", "F", ""],
                    "x": [1, 1, 1, 1, 2, 2, 3],
                }
            ),
            indexes=["x"],
        )

        # Run op
        op = Resample(input=evset.node(), sampling=sampling_evset.node())
        op.outputs["output"].check_same_sampling(sampling_evset.node())
        instance = ResampleNumpyImplementation(op)
        output = instance.call(input=evset, sampling=sampling_evset)["output"]

        print("output:\n", output)
        print("expected_output:\n", expected_output)

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
