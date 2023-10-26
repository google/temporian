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
import math
import numpy as np
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, f64


class UnaryTest(absltest.TestCase):
    """Test numpy implementation of all unary operators"""

    def test_invert_boolean(self) -> None:
        """Test inversion of boolean features"""
        evset = event_set(
            timestamps=[0, 1, 2, 1],
            features={
                "store_id": [1, 1, 1, 2],
                "product_id": [1, 2, 2, 1],
                "bool_1": [True, True, False, False],
                "bool_2": [True, False, True, False],
            },
            indexes=["store_id", "product_id"],
        )
        expected = event_set(
            timestamps=[0, 1, 2, 1],
            features={
                "store_id": [1, 1, 1, 2],
                "product_id": [1, 2, 2, 1],
                "bool_1": [False, False, True, True],
                "bool_2": [False, True, False, True],
            },
            indexes=["store_id", "product_id"],
            same_sampling_as=evset,
        )
        assertOperatorResult(self, ~evset, expected)

    def test_error_nonboolean(self) -> None:
        """Check that trying to invert a non-boolean raises ValueError"""
        evset = event_set(timestamps=[1, 2, 3], features={"f": [1, 2, 3]})

        with self.assertRaisesRegex(ValueError, "bool"):
            _ = ~evset

    def test_correct_abs(self) -> None:
        evset = event_set(timestamps=[1, 2, 3], features={"f": [1, -2, -3]})
        expected = event_set(
            timestamps=[1, 2, 3],
            features={"f": [1, 2, 3]},
            same_sampling_as=evset,
        )
        assertOperatorResult(self, evset.abs(), expected)
        assertOperatorResult(self, abs(evset), expected)  # __abs__ magic

    def test_correct_log(self) -> None:
        evset = event_set(
            timestamps=[1, 2, 3, 4], features={"f": [1, np.e, 0, 10]}
        )
        expected = event_set(
            timestamps=[1, 2, 3, 4],
            features={"f": [0, 1, -np.inf, np.log(10)]},
            same_sampling_as=evset,
        )
        assertOperatorResult(self, evset.log(), expected)

    def test_correct_isnan(self) -> None:
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"f": f64([1, 0, np.nan, math.nan, None])},
        )
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"f": [False, False, True, True, True]},
            same_sampling_as=evset,
        )
        assertOperatorResult(self, evset.isnan(), expected)

    def test_correct_notnan(self) -> None:
        evset = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"f": f64([1, 0, np.nan, math.nan, None])},
        )
        expected = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={"f": [True, True, False, False, False]},
            same_sampling_as=evset,
        )
        assertOperatorResult(self, evset.notnan(), expected)


if __name__ == "__main__":
    absltest.main()
