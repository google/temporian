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


class EndTest(TestCase):
    def test_basic(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features={"a": [5, 6, 7, 8], "b": ["A", "A", "B", "B"]},
            indexes=["b"],
        )

        result = evset.end()

        expected = event_set(
            timestamps=[2, 4], features={"b": ["A", "B"]}, indexes=["b"]
        )

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_empty(self):
        evset = event_set(timestamps=[], features={"a": []})

        result = evset.end()

        expected = event_set(timestamps=[])

        assertOperatorResult(self, result, expected, check_sampling=False)


if __name__ == "__main__":
    absltest.main()
