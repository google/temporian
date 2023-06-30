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

import numpy as np
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
        evset_1 = event_set(
            timestamps=[1, 2, 3],
            features={"a": [5, 6, 7]},
        )
        evset_2 = event_set(
            timestamps=[1, 2, 4],
            features={"b": [8, 9, 10]},
        )

        expected_output = event_set(
            timestamps=[1, 2, 3],
            features={
                "a": [5, 6, 7],
                "b": [8, 9, math.nan],
            },
        )

        # Run op
        op = Join(input_1=evset_1.node(), input_2=evset_2.node())
        instance = JoinNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input_1=evset_1, input_2=evset_2)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_base_on(self):
        evset_1 = event_set(
            timestamps=[1, 2, 2, 3],
            features={
                "a": [5, 6, 7, 8],
                "c": [0, 1, 2, 3],
            },
        )
        evset_2 = event_set(
            timestamps=[1, 2, 2, 3],
            features={
                "b": [5, 6, 7, 8],
                "c": [0, 2, 1, 3],
            },
        )

        expected_output = event_set(
            timestamps=[1, 2, 2, 3],
            features={
                "a": [5, 6, 7, 8],
                "b": [5, 7, 6, 8],
                "c": [0, 1, 2, 3],
            },
        )

        # Run op
        op = Join(input_1=evset_1.node(), input_2=evset_2.node(), on="c")
        instance = JoinNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input_1=evset_1, input_2=evset_2)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
