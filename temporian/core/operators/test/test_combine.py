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

import numpy as np
from temporian.core.operators.combine import How, combine
from temporian.implementation.numpy.data.io import event_set

from temporian.test.utils import assertOperatorResult


class CombineTest(absltest.TestCase):
    def setUp(self):
        # Indexes: a, b, c
        self.evset_1 = event_set(
            timestamps=[0, 1, 2, 3, 4, 5],
            features={
                "a": [0, 1, 2, 3, 4, 5],
                "idx": ["a", "a", "b", "b", "c", "c"],
            },
            indexes=["idx"],
        )
        # Indexes: b, c, d
        self.evset_2 = event_set(
            timestamps=[0, 1, 2, 3, 4, 5],
            features={
                "a": [0, 1, 2, 3, 4, 5],
                "idx": ["b", "b", "c", "c", "d", "d"],
            },
            indexes=["idx"],
        )
        # Indexes: c, d, e
        self.evset_3 = event_set(
            timestamps=[0, 1, 2, 3, 4, 5],
            features={
                "a": [0, 1, 2, 3, 4, 5],
                "idx": ["c", "c", "d", "d", "e", "e"],
            },
            indexes=["idx"],
        )

    def test_combine_left(self):
        # left mode (only use index values from left input -> a,b,c)
        expected_output = event_set(
            timestamps=[0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
            features={
                "a": [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
                "idx": [
                    # a is only present in input 1
                    "a",
                    "a",
                    # b is present in inputs 1,2
                    "b",
                    "b",
                    "b",
                    "b",
                    # c is present in inputs 1,2,3
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                ],
            },
            indexes=["idx"],
        )
        # NOTE: check_sampling=False still checks timestamps
        assertOperatorResult(
            self,
            combine(self.evset_1, self.evset_2, self.evset_3, how=How.left),
            expected_output,
            check_sampling=False,
        )

    def test_combine_inner(self):
        # inner mode (only c indexes survive)
        expected_output = event_set(
            timestamps=[0, 1, 2, 3, 4, 5],
            features={
                "a": [0, 1, 2, 3, 4, 5],
                "idx": ["c", "c", "c", "c", "c", "c"],
            },
            indexes=["idx"],
        )

        assertOperatorResult(
            self,
            combine(self.evset_1, self.evset_2, self.evset_3, how=How.inner),
            expected_output,
            check_sampling=False,
        )

    def test_combine_outer(self):
        # outer mode
        expected_output = event_set(
            timestamps=[0, 1, 2, 3, 4, 5] * 3,
            features={
                "a": [0, 1, 2, 3, 4, 5] * 3,
                "idx": (
                    ["a", "a", "b", "b", "c", "c"]  # input 1
                    + ["b", "b", "c", "c", "d", "d"]  # input 2
                    + ["c", "c", "d", "d", "e", "e"]
                ),  # input 3
            },
            indexes=["idx"],
        )

        assertOperatorResult(
            self,
            combine(self.evset_1, self.evset_2, self.evset_3, how=How.outer),
            expected_output,
            check_sampling=False,
        )

    def test_combine_two_noindex(self):
        evset_1 = event_set(
            timestamps=[0, 1, 2, 2.5, 3, 4],
            features={
                "a": [0.0, -10.0, -20.0, -25.0, -30.0, -40.0],
                "b": [0, 10, 20, 25, 30, 40],
            },
        )
        # Features are in different order (should use first input's order)
        evset_2 = event_set(
            timestamps=[-2, -1, 0, 3, 8],
            features={
                "b": [-20, -10, 0, 30, 80],
                "a": [20.0, 10.0, 0.0, -30.0, -80.0],
            },
        )
        expected_output = event_set(
            timestamps=[-2, -1, 0, 0, 1, 2, 2.5, 3, 3, 4, 8],
            features={
                "a": [20.0, 10, 0, 0, -10, -20, -25, -30, -30, -40, -80],
                "b": [-20, -10, 0, 0, 10, 20, 25, 30, 30, 40, 80],
            },
        )
        assertOperatorResult(
            self,
            combine(evset_1, evset_2, how=How.outer),
            expected_output,
            check_sampling=False,
        )

    def test_combine_multiple_indexed(self):
        evsets_list = []
        base_timestamps = np.array([0, 1, 2, -10, 0, 10])
        base_fa = np.array([55, 23, 44, 0, 31, 66])
        base_fb = ["a", "b", "c", "d", "e", "f"]
        indexes = ["A", "A", "A", "B", "B", "B"]
        n = 5
        for i in range(n):
            evsets_list.append(
                event_set(
                    timestamps=base_timestamps + 0.01 * i,
                    features={
                        "a": base_fa + i,
                        "b": base_fb,
                        "idx": indexes,
                    },
                    indexes=["idx"],
                )
            )
        expected_output = event_set(
            timestamps=[
                b + 0.01 * i for i in range(n) for b in base_timestamps
            ],
            features={
                "a": [b + i for i in range(n) for b in base_fa],
                "b": [b for _ in range(n) for b in base_fb],
                "idx": [idx for _ in range(n) for idx in indexes],
            },
            indexes=["idx"],
        )
        assertOperatorResult(
            self,
            combine(*evsets_list, how=How.outer),
            expected_output,
            check_sampling=False,
        )

    def test_combine_error_different_feature_names(self):
        # Test error msg when two events have different feature names
        evset_1 = event_set(
            timestamps=[1.0],
            features={
                "a": [0],
                "b": [0.0],
            },
        )
        evset_2 = event_set(
            timestamps=[1.0],
            features={
                "a": [0],
                "c": [0.0],
            },
        )
        with self.assertRaisesRegex(ValueError, "features are different"):
            combine(evset_1, evset_2, how=How.outer)

    def test_combine_error_different_dtypes(self):
        # Same feature names, but second evset has a float feature
        evset_1 = event_set(
            timestamps=[1.0],
            features={
                "a": [0],
                "b": [0],
            },
        )
        evset_2 = event_set(
            timestamps=[1.0],
            features={
                "a": [0],
                "b": [0.0],
            },
        )
        # Run op
        with self.assertRaisesRegex(ValueError, "features are different"):
            combine(evset_1, evset_2)

    def test_combine_error_noncompatible_indexes(self):
        # Different index names
        evset_1 = event_set(
            timestamps=[1.0, 1.0],
            features={"a": [0, 1], "idx1": ["A", "C"]},
            indexes=["idx1"],
        )
        evset_2 = event_set(
            timestamps=[1.0, 1.0],
            features={"a": [0, 1], "idx2": ["A", "C"]},
            indexes=["idx2"],
        )
        with self.assertRaisesRegex(
            ValueError, "Arguments don't have the same index"
        ):
            combine(evset_1, evset_2)

    def test_combine_how_enum(self):
        # Valid options
        combine(self.evset_1, self.evset_2, how="inner")
        combine(self.evset_1, self.evset_2, how="outer")
        combine(self.evset_1, self.evset_2, how="left")

        with self.assertRaisesRegex(ValueError, "Invalid argument"):
            combine(self.evset_1, self.evset_2, how="invalid")


if __name__ == "__main__":
    absltest.main()
