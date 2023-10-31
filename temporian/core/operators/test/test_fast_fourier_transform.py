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
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, f32, i32


class FastFourierTransformTest(TestCase):
    @parameterized.parameters(
        {"window": None, "num_spectral_lines": None},
        {"window": "hamming", "num_spectral_lines": None},
        {"window": None, "num_spectral_lines": 1},
    )
    def test_base(self, window, num_spectral_lines):
        x = [0, 1, 2, 3, 4, 5]
        y = f32([0, 1, 5, -1, 3, 1])

        def expected(data, frequency_idx):
            if window == "hamming":
                data = np.hamming(len(data)) * data
            else:
                assert window is None
            return np.abs(np.fft.fft(data)).astype(np.float32)[frequency_idx]

        evset = event_set(timestamps=x, features={"a": y})

        if num_spectral_lines is None:
            num_spectral_lines = 2

        expected_features = {}
        for i in range(num_spectral_lines):
            expected_features[f"a{i}"] = [
                expected([0, 1, 5, -1], i),
                expected([1, 5, -1, 3], i),
                expected([5, -1, 3, 1], i),
            ]

        expected_output = event_set(
            timestamps=x[3:], features=expected_features
        )

        result = evset.experimental_fast_fourier_transform(
            num_events=4,
            hop_size=1,
            window=window,
            num_spectral_lines=num_spectral_lines,
        )

        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    def test_good(self):
        evset = event_set([0, 0, 0, 0], features={"a": f32([0, 0, 0, 0])})
        evset.experimental_fast_fourier_transform(num_events=4)
        evset.experimental_fast_fourier_transform(
            num_events=4, window="hamming"
        )
        evset.experimental_fast_fourier_transform(
            num_events=4, window="hamming", num_spectral_lines=2
        )

    def test_wrong_dtype(self):
        evset = event_set([], {"a": i32([])})
        with self.assertRaisesRegex(ValueError, "should be tp.float32"):
            evset.experimental_fast_fourier_transform(num_events=20)

    def test_wrong_features(self):
        evset = event_set([], {"a": f32([]), "b": f32([])})
        with self.assertRaisesRegex(ValueError, "to be a single feature"):
            evset.experimental_fast_fourier_transform(num_events=20)

    def test_wrong_num_events(self):
        evset = event_set([], {"a": f32([])})
        with self.assertRaisesRegex(ValueError, "should be strictly positive"):
            evset.experimental_fast_fourier_transform(num_events=0)

    def test_wrong_window(self):
        evset = event_set([], {"a": f32([])})
        with self.assertRaisesRegex(ValueError, "window should be None or"):
            evset.experimental_fast_fourier_transform(
                num_events=20, window="AAA"
            )

    def test_wrong_num_spectral_lines(self):
        evset = event_set([], {"a": f32([])})
        with self.assertRaisesRegex(
            ValueError, "num_spectral_lines should be less or equal"
        ):
            evset.experimental_fast_fourier_transform(
                num_events=20, num_spectral_lines=15
            )


if __name__ == "__main__":
    absltest.main()
