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
from temporian.core.operators.combine import Combine
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.combine import (
    CombineNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class CombineOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_combine_two_noindex(self):
        evset_1 = event_set(
            timestamps=[0, 1, 2, 2.5, 3, 4],
            features={
                "a": [0.0, -10.0, -20.0, -25.0, -30.0, -40.0],
                "b": [0, 10, 20, 25, 30, 40],
            },
        )
        evset_2 = event_set(
            timestamps=[-2, -1, 0, 3, 8],
            features={
                "b": [-20, -10, 0, 30, 80],
                "a": [20.0, 10.0, 0.0, -30.0, -80.0],
            },
        )
        node_1 = evset_1.node()
        node_2 = evset_2.node()

        expected_output = event_set(
            timestamps=[-2, -1, 0, 0, 1, 2, 2.5, 3, 3, 4, 8],
            features={
                "a": [20.0, 10, 0, 0, -10, -20, -25, -30, -30, -40, -80],
                "b": [-20, -10, 0, 0, 10, 20, 25, 30, 30, 40, 80],
            },
        )

        # Run op
        op = Combine(input_0=node_1, input_1=node_2)
        instance = CombineNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input_0=evset_1, input_1=evset_2)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_combine_multiple_indexed(self):
        nodes_dict = {}
        evsets_dict = {}
        base_timestamps = np.array([0, 1, 2, -10, 0, 10])
        base_fa = np.array([55, 23, 44, 0, 31, 66])
        base_fb = ["a", "b", "c", "d", "e", "f"]
        indexes = ["A", "A", "A", "B", "B", "B"]
        n = 5
        for i in range(n):
            evset = event_set(
                timestamps=base_timestamps + 0.01 * i,
                features={
                    "a": base_fa + i,
                    "b": base_fb,
                    "idx": indexes,
                },
                indexes=["idx"],
            )
            evsets_dict[f"input_{i}"] = evset
            nodes_dict[f"input_{i}"] = evset.node()
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

        # Run op
        op = Combine(**nodes_dict)
        instance = CombineNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(**evsets_dict)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_combine_different_feature_names(self):
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
        node_1 = evset_1.node()
        node_2 = evset_2.node()

        # Run op
        with self.assertRaisesRegex(ValueError, "features are different"):
            Combine(input_0=node_1, input_1=node_2)

    def test_combine_different_dtypes(self):
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
        node_1 = evset_1.node()
        node_2 = evset_2.node()

        # Run op
        with self.assertRaisesRegex(ValueError, "features are different"):
            Combine(input_0=node_1, input_1=node_2)

    def test_combine_noncompatible_indexes(self):
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
        node_1 = evset_1.node()
        node_2 = evset_2.node()

        # Run op
        with self.assertRaisesRegex(
            ValueError, "Arguments don't have the same index"
        ):
            Combine(input_0=node_1, input_1=node_2)


if __name__ == "__main__":
    absltest.main()
