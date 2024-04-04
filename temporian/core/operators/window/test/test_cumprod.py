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
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class CumprodTest(TestCase):
    def test_basic(self):
        evset = event_set(
            timestamps=[1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            features={
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
                "a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                "b": [1.0, -1.0, 2.0, -3.0, -8.0, 0.0, 5.0, 3.0, -1.0],
            },
            indexes=["x", "y"],
        )

        expected = event_set(
            timestamps=[1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            features={
                "x": ["X1", "X1", "X1", "X2", "X2", "X2", "X2", "X2", "X2"],
                "y": ["Y1", "Y1", "Y1", "Y1", "Y1", "Y1", "Y2", "Y2", "Y2"],
                "a": [
                    10.0,
                    110.0,
                    1320.0,
                    13.0,
                    182.0,
                    2730.0,
                    16.0,
                    272.0,
                    4896.0,
                ],
                "b": [1.0, -1.0, -2.0, -3.0, 24.0, 0, 5.0, 15.0, -15.0],
            },
            indexes=["x", "y"],
            same_sampling_as=evset,
        )

        result = evset.cumprod()
        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
