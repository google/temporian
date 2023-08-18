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

import numpy as np

from temporian.core.data.node import input_node
from temporian.core.data.dtype import DType
from temporian.core.operators.fast_fourier_transform import (
    fast_fourier_transform,
)


class FFTOperatorTest(absltest.TestCase):
    def test_good(self):
        input = input_node([("a", DType.FLOAT32)])
        fast_fourier_transform(input, num_events=20)
        fast_fourier_transform(input, num_events=20, window="hamming")
        fast_fourier_transform(
            input, num_events=20, window="hamming", num_spectral_lines=10
        )

    def test_wrong_dtype(self):
        input = input_node([("a", DType.INT32)])
        with self.assertRaisesRegex(ValueError, "should be tp.float32"):
            fast_fourier_transform(input, num_events=20)

    def test_wrong_features(self):
        input = input_node([("a", DType.FLOAT32), ("b", DType.FLOAT32)])
        with self.assertRaisesRegex(ValueError, "to be a single feature"):
            fast_fourier_transform(input, num_events=20)

    def test_wrong_num_events(self):
        input = input_node([("a", DType.FLOAT32)])
        with self.assertRaisesRegex(ValueError, "should be strictly positive"):
            fast_fourier_transform(input, num_events=0)

    def test_wrong_window(self):
        input = input_node([("a", DType.FLOAT32)])
        with self.assertRaisesRegex(ValueError, "window should be None or"):
            fast_fourier_transform(input, num_events=20, window="AAA")

    def test_wrong_num_spectral_lines(self):
        input = input_node([("a", DType.FLOAT32)])
        with self.assertRaisesRegex(
            ValueError, "num_spectral_lines should be less or equal"
        ):
            fast_fourier_transform(input, num_events=20, num_spectral_lines=15)


if __name__ == "__main__":
    absltest.main()
