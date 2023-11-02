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

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, i64


class EnumerateTest(absltest.TestCase):
    def test_base(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 0, 1],
            features={
                "a": [1.0, 2.0, 3.0, 4.0, 0.0, 1.0],
                "b": [5, 6, 7, 8, 1, 2],
                "c": ["A", "A", "A", "A", "B", "B"],
            },
            indexes=["c"],
        )

        expected_output = event_set(
            timestamps=[1, 2, 3, 4, 0, 1],
            features={
                "enumerate": [0, 1, 2, 3, 0, 1],
                "c": ["A", "A", "A", "A", "B", "B"],
            },
            indexes=["c"],
            same_sampling_as=evset,
        )
        assertOperatorResult(self, evset.enumerate(), expected_output)

    def test_empty(self):
        evset = event_set(timestamps=[])
        expect_evset = event_set(
            timestamps=[],
            features={"enumerate": i64([])},
            same_sampling_as=evset,
        )
        assertOperatorResult(self, evset.enumerate(), expect_evset)


if __name__ == "__main__":
    absltest.main()
