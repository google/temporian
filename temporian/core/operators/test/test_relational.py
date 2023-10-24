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


class RelationalTest(absltest.TestCase):
    """Test relational operators: equal, greater, greater or equal, etc."""

    def setUp(self):
        self.evset_1 = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "store_id": [0, 0, 1, 1, 2, 2],
                "sales": [10.0, 1.0, 12.0, np.nan, -30.0, 0],
            },
            indexes=["store_id"],
        )
        self.evset_2 = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "store_id": [0, 0, 1, 1, 2, 2],
                "cost": [10.0, -1.0, 12.000000001, np.nan, -30.0, np.nan],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset_1,
        )

    def test_correct_equal(self) -> None:
        """Test correct equal operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "store_id": [0, 0, 1, 1, 2, 2],
                # NOTE: nan == nan is False
                "eq_sales_cost": [True, False, False, False, True, False],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(self, self.evset_1.equal(self.evset_2), expected)

    def test_equal_str(self) -> None:
        timestamps = [1, 2, 3, 4]
        evset_1 = event_set(
            timestamps=timestamps,
            features={"a": ["A", "A", "B", "B"]},
        )
        evset_2 = event_set(
            timestamps=timestamps,
            features={"b": ["A", "B", "A", "B"]},
            same_sampling_as=evset_1,
        )

        expected = event_set(
            timestamps=timestamps,
            features={"eq_a_b": [True, False, False, True]},
            same_sampling_as=evset_1,
        )
        # NOTE: evset_1 == evset_2 is not overloaded for event-wise comparison
        assertOperatorResult(self, evset_1.equal(evset_2), expected)

    def test_notequal_str(self) -> None:
        timestamps = [1, 2, 3, 4]
        evset_1 = event_set(
            timestamps=timestamps,
            features={"a": ["A", "A", "B", "B"]},
        )
        evset_2 = event_set(
            timestamps=timestamps,
            features={"b": ["A", "B", "A", "B"]},
            same_sampling_as=evset_1,
        )

        expected = event_set(
            timestamps=timestamps,
            features={"ne_a_b": [False, True, True, False]},
            same_sampling_as=evset_1,
        )

        assertOperatorResult(self, evset_1 != evset_2, expected)

    def test_correct_not_equal(self) -> None:
        """Test correct not-equal operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "store_id": [0, 0, 1, 1, 2, 2],
                # NOTE: nan != nan is True
                "ne_sales_cost": [False, True, True, True, False, True],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(self, self.evset_1 != self.evset_2, expected)

    def test_correct_greater(self) -> None:
        """Test '>' operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "store_id": [0, 0, 1, 1, 2, 2],
                # NOTE: any comparison to nan is always False
                "gt_sales_cost": [False, True, False, False, False, False],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(self, self.evset_1 > self.evset_2, expected)

    def test_correct_greater_equal(self) -> None:
        """Test '>=' operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "store_id": [0, 0, 1, 1, 2, 2],
                # NOTE: any comparison to nan is always False
                "ge_sales_cost": [True, True, False, False, True, False],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(self, self.evset_1 >= self.evset_2, expected)

    def test_correct_less(self) -> None:
        """Test '<' operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "store_id": [0, 0, 1, 1, 2, 2],
                # NOTE: any comparison to nan is always False
                "lt_sales_cost": [False, False, True, False, False, False],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(self, self.evset_1 < self.evset_2, expected)

    def test_correct_less_equal(self) -> None:
        """Test '<=' operator."""
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5, 6],
            features={
                "store_id": [0, 0, 1, 1, 2, 2],
                # NOTE: any comparison to nan is always False
                "le_sales_cost": [True, False, True, False, True, False],
            },
            indexes=["store_id"],
            same_sampling_as=self.evset_1,
        )
        assertOperatorResult(self, self.evset_1 <= self.evset_2, expected)


if __name__ == "__main__":
    absltest.main()
