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
from temporian.test.utils import assertOperatorResult


class TickTest(parameterized.TestCase):
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

    @parameterized.parameters(
        # Not aligned, end matches last tick
        {
            "input": [1, 5, 9],
            "output": [1, 5, 9],
            "align": False,
            "interval": 4,
        },
        # Not aligned, end doesn't match last tick
        {
            "input": [1, 5, 9],
            "output": [1, 4, 7],
            "align": False,
            "interval": 3,
        },
        # Aligned, end doesn't match last tick
        {
            "input": [1, 5, 9],
            "output": [4, 8],
            "align": True,
            "interval": 4,
        },
        # Aligned, end matches last tick
        {
            "input": [1, 5, 9],
            "output": [3, 6, 9],
            "align": True,
            "interval": 3,
        },
    )
    def test_not_inclusive_right(self, input, output, align, interval):
        evset = event_set(input)
        result = evset.tick(interval, align=align, include_right=False)
        expected_output = event_set(output)
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    @parameterized.parameters(
        # Not aligned, end matches last tick
        {
            "input": [1, 5, 9],
            "output": [1, 5, 9],
            "align": False,
            "interval": 4,
        },
        # Not aligned, end doesn't match last tick
        {
            "input": [1, 5, 9],
            "output": [1, 4, 7, 10],
            "align": False,
            "interval": 3,
        },
        # Aligned, end doesn't match last tick
        {
            "input": [1, 5, 9],
            "output": [4, 8, 12],
            "align": True,
            "interval": 4,
        },
        # Aligned, end matches last tick
        {
            "input": [1, 5, 9],
            "output": [3, 6, 9],
            "align": True,
            "interval": 3,
        },
    )
    def test_inclusive_right(self, input, output, align, interval):
        evset = event_set(input)
        result = evset.tick(interval, align=align, include_right=True)
        expected_output = event_set(output)
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    @parameterized.parameters(
        # Not aligned, start matches first tick
        {
            "input": [1, 5, 9],
            "output": [1, 5, 9],
            "align": False,
            "interval": 4,
        },
        # Aligned, start doesn't match first tick
        {
            "input": [1, 5, 9],
            "output": [4, 8, 12],
            "align": True,
            "interval": 4,
        },
    )
    def test_not_inclusive_left(self, input, output, align, interval):
        evset = event_set(input)
        result = evset.tick(interval, align=align, include_left=False)
        expected_output = event_set(output)
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    @parameterized.parameters(
        # Not aligned, start matches first tick
        {
            "input": [1, 5, 9],
            "output": [1, 5, 9],
            "align": False,
            "interval": 4,
        },
        # Aligned, start doesn't match first tick
        {
            "input": [1, 5, 9],
            "output": [0, 4, 8, 12],
            "align": True,
            "interval": 4,
        },
    )
    def test_inclusive_left(self, input, output, align, interval):
        evset = event_set(input)
        result = evset.tick(interval, align=align, include_left=True)
        expected_output = event_set(output)
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )

    @parameterized.parameters(
        # all negatives
        {
            "input": [-9, -5, -1],
            "output": [-9, -5, -1],
            "align": False,
            "interval": 4,
            "right": False,
            "left": False,
        },
        # include right, end matches last tick
        {
            "input": [-9, -5, -1],
            "output": [-9, -5, -1],
            "align": False,
            "interval": 4,
            "right": True,
            "left": False,
        },
        # include right, end doesn't match last tick
        {
            "input": [-9, -5, -1],
            "output": [-9, -6, -3, 0],
            "align": False,
            "interval": 3,
            "right": True,
            "left": False,
        },
        # include left, include left
        {
            "input": [-9, -5, -1],
            "output": [-9, -5, -1],
            "align": False,
            "interval": 4,
            "right": False,
            "left": True,
        },
        # include left, aligned so that left doesn't match first tick
        {
            "input": [-9, -5, -1],
            "output": [-12, -8, -4],
            "align": True,
            "interval": 4,
            "right": False,
            "left": True,
        },
        # include left adn right, aligned so that left doesn't match first tick
        {
            "input": [-9, -5, -1],
            "output": [-12, -8, -4, 0],
            "align": True,
            "interval": 4,
            "right": True,
            "left": True,
        },
    )
    def test_negatives(self, input, output, align, interval, right, left):
        evset = event_set(input)
        result = evset.tick(
            interval, align=align, include_right=right, include_left=left
        )
        expected_output = event_set(output)
        assertOperatorResult(
            self, result, expected_output, check_sampling=False
        )


if __name__ == "__main__":
    absltest.main()
