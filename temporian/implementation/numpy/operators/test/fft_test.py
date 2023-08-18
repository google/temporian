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


from absl.testing import parameterized, absltest

import numpy as np
from temporian.core.operators.fast_fourier_transform import FastFourierTransform
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.fast_fourier_transform import (
    FastFourierTransformNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class FastFourierTransformOperatorTest(parameterized.TestCase):
    def setUp(self):
        pass

    @parameterized.parameters(
        {"window": None, "num_spectral_lines": None},
        {"window": "hamming", "num_spectral_lines": None},
        {"window": None, "num_spectral_lines": 1},
    )
    def test_base(self, window, num_spectral_lines):
        x = [0, 1, 2, 3, 4, 5]
        y = np.array([0, 1, 5, -1, 3, 1], dtype=np.float32)

        def expected(data, frequency_idx):
            if window == "hamming":
                data = np.hamming(len(data)) * data
            else:
                assert window is None
            return np.abs(np.fft.fft(data)).astype(np.float32)[frequency_idx]

        evset = event_set(timestamps=x, features={"a": y})
        node = evset.node()

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
            timestamps=x[3:],
            features=expected_features,
        )

        # Run op
        op = FastFourierTransform(
            input=node,
            num_events=4,
            hop_size=1,
            window=window,
            num_spectral_lines=num_spectral_lines,
        )
        instance = FastFourierTransformNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
