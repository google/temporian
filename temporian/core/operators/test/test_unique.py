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


from absl.testing import absltest, parameterized

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, f64, i32


class UniqueTest(parameterized.TestCase):
    def test_empty(self):
        evset = event_set(timestamps=[], features={"a": i32([])})
        result = evset.unique()
        expected = event_set(timestamps=[], same_sampling_as=evset)

        assertOperatorResult(self, result, expected, check_sampling=True)

    @parameterized.parameters(
        {"in_timestamps": [0.0, 1.0], "out_timestamps": [0.0, 1.0]},
        {"in_timestmaps": [1.0, 2.0], "out_timestamps": [1.0, 2.0]},
    )
    def test_base(self, in_timestamps, out_timestamps):
        evset = event_set(
            timestamps=in_timestamps,
            features={
                "a": f64([0, 0]),
            },
        )
        result = evset.unique()
        expected = event_set(
            timestamps=out_timestamps,
            features={
                "a": f64([0, 0]),
            },
        )

        assertOperatorResult(self, result, expected, check_sampling=False)


if __name__ == "__main__":
    absltest.main()

