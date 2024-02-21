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
        expected = event_set([1, 5, 9])

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
        expected_output = event_set([4, 8, 12])
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
        expected_output = event_set([])
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

    def test_not_inclusive_right(self):
        evset = event_set([1, 5, 9])
        # Not aligned, no include right, end matches last tick
        result = evset.tick(4.0, align=False, include_right=False)
        expected_output = event_set([1, 5, 9])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )
        # Not aligned, no include right, end doesn't match last tick
        result = evset.tick(3.0, align=False, include_right=False)
        expected_output = event_set([1, 4, 7])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

        # Aligned, no include right, end doesn't match last tick
        result = evset.tick(4.0, align=True, include_right=False)
        expected_output = event_set([4, 8])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )
        # Not aligned, no include right, end matches last tick
        result = evset.tick(3.0, align=True, include_right=False)
        expected_output = event_set([3, 6, 9])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    def test_inclusive_right(self):
        evset = event_set([1, 5, 9])
        # Not aligned, include right, end matches last tick
        result = evset.tick(4.0, align=False, include_right=True)
        expected_output = event_set([1, 5, 9])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )
        # Not aligned, include right, end doesn't match last tick
        result = evset.tick(3.0, align=False, include_right=True)
        expected_output = event_set([1, 4, 7, 10])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

        # Aligned, include right, end doesn't match last tick
        result = evset.tick(4.0, align=True, include_right=True)
        expected_output = event_set([4, 8, 12])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )
        # Not aligned, include right, end matches last tick
        result = evset.tick(3.0, align=True, include_right=True)
        expected_output = event_set([3, 6, 9])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    def test_not_inclusive_left(self):
        evset = event_set([1, 5, 9])
        # Not aligned, no include left, start matches first tick
        result = evset.tick(4.0, align=False, include_left=False)
        expected_output = event_set([1, 5, 9])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

        # Aligned, no include left, start doesn't match first tick
        result = evset.tick(4.0, align=True, include_left=False)
        expected_output = event_set([4, 8, 12])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    def test_inclusive_left(self):
        evset = event_set([1, 5, 9])
        # Not aligned, include left, start matches first tick
        result = evset.tick(4.0, align=False, include_left=True)
        expected_output = event_set([1, 5, 9])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

        # Aligned, include left, start doesn't match first tick
        result = evset.tick(4.0, align=True, include_left=True)
        expected_output = event_set([0, 4, 8, 12])
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )


if __name__ == "__main__":
    absltest.main()
