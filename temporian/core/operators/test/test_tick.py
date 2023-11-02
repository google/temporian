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


class TickTest(TestCase):
    def test_no_align(self):
        evset = event_set([1, 5.5])
        result = evset.tick(4.0, align=False)
        expected = event_set([1, 5])

        assertOperatorResult(self, result, expected, check_sampling=False)

    def test_on_the_spot(self):
        evset = event_set([0, 4, 8])
        result = evset.tick(4.0, align=False)
        expected_output = event_set([0, 4, 8])

        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    def test_align(self):
        evset = event_set([1, 5.5, 8.1])
        expected_output = event_set([4, 8])
        result = evset.tick(4.0, align=True)

        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    def test_empty(self):
        evset = event_set([])
        expected_output = event_set([])
        result = evset.tick(4.0, align=False)

        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    def test_no_align_single(self):
        evset = event_set([1])
        expected_output = event_set([1])
        result = evset.tick(4.0, align=False)

        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    def test_align_single(self):
        evset = event_set([1])
        expected_output = event_set([])
        result = evset.tick(4.0, align=True)

        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )


if __name__ == "__main__":
    absltest.main()
