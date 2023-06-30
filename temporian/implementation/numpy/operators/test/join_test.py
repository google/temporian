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

from absl.testing import absltest

from temporian.core.operators.join import Join
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.join import (
    JoinNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class JoinOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        evset_left = event_set(
            timestamps=[1, 2, 3, 5, 5, 6, 6],
            features={"a": [11, 12, 13, 14, 15, 16, 17]},
        )
        evset_right = event_set(
            timestamps=[1, 2, 4, 5, 5],
            features={"b": [21.0, 22.0, 23.0, 24.0, 25.0]},
        )

        expected_output = event_set(
            timestamps=[1, 2, 3, 5, 5, 6, 6],
            features={
                "a": [11, 12, 13, 14, 15, 16, 17],
                "b": [21.0, 22.0, math.nan, 24.0, 24.0, math.nan, math.nan],
            },
        )

        # Run op
        op = Join(left=evset_left.node(), right=evset_right.node())
        instance = JoinNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(left=evset_left, right=evset_right)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_base_on(self):
        evset_left = event_set(
            timestamps=[1, 2, 2, 3, 4, 5],
            features={
                "a": [11, 12, 13, 14, 15, 16],
                "c": [0, 1, 2, 3, 4, 5],
            },
        )
        evset_right = event_set(
            timestamps=[1, 2, 2, 3, 4],
            features={
                "c": [0, 2, 1, 3, 5],
                "b": [11.0, 12.0, 13.0, 14.0, 15.0],
            },
        )

        expected_output = event_set(
            timestamps=[1, 2, 2, 3, 4, 5],
            features={
                "a": [11, 12, 13, 14, 15, 16],
                "c": [0, 1, 2, 3, 4, 5],
                "b": [11.0, 13.0, 12.0, 14.0, math.nan, math.nan],
            },
        )

        # Run op
        op = Join(left=evset_left.node(), right=evset_right.node(), on="c")
        instance = JoinNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(left=evset_left, right=evset_right)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
