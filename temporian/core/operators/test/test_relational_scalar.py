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
from absl.testing import absltest
from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class RelationalScalarTest(absltest.TestCase):
    """Test implementation of arithmetic operators with a scalar value:
    addition, subtraction, division and multiplication"""

    def setUp(self):
        self.evset = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 1],
                "sales": [10.0, 0.0, 12.0, np.nan, 30.0],
            },
            indexes=["store_id"],
        )

    def test_correct_equal(self) -> None:
        """Test equal operator."""
        value = 12.0

        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 1],
                "sales": [False, False, True, False, False],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset.equal(value), expected)

    def test_equal_str(self) -> None:
        value = "A"

        timestamps = [1, 2, 3, 4]
        evset = event_set(
            timestamps=timestamps,
            features={"a": ["A", "A", "B", "B"]},
        )

        expected = event_set(
            timestamps=timestamps,
            features={"a": [True, True, False, False]},
            same_sampling_as=evset,
        )

        assertOperatorResult(self, evset.equal(value), expected)

    def test_notequal_str(self) -> None:
        value = "A"
        timestamps = [1, 2, 3, 4]
        evset = event_set(
            timestamps=timestamps,
            features={"a": ["A", "A", "B", "B"]},
        )

        expected = event_set(
            timestamps=timestamps,
            features={"a": [False, False, True, True]},
            same_sampling_as=evset,
        )
        assertOperatorResult(self, evset != value, expected)

    def test_equal_nan(self) -> None:
        """Test equal operator against a nan value."""
        # NOTE: any comparison to nan should be False, even nan==nan
        value = np.nan

        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 1],
                "sales": [False, False, False, False, False],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset.equal(value), expected)

    def test_greater_scalar(self) -> None:
        value = 11
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 1],
                "sales": [False, False, True, False, True],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset >= value, expected)

    def test_less_scalar(self) -> None:
        value = 11
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 1],
                "sales": [True, True, False, False, False],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset < value, expected)

    def test_greater_equal_scalar(self) -> None:
        value = 12
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 1],
                "sales": [False, False, True, False, True],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset >= value, expected)

    def test_less_equal_scalar(self) -> None:
        value = 12
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 1],
                "sales": [True, True, True, False, False],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset <= value, expected)

    def test_not_equal_scalar(self) -> None:
        value = 12
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 1],
                "sales": [True, True, False, True, True],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset != value, expected)

    def test_not_equal_nan(self) -> None:
        # a != nan should be True always (even if a=np.nan)
        value = np.nan
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "store_id": [0, 0, 1, 1, 1],
                "sales": [True, True, True, True, True],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset,
        )
        assertOperatorResult(self, self.evset != value, expected)


if __name__ == "__main__":
    absltest.main()
