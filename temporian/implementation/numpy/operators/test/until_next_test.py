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

import math

import numpy as np
from temporian.core.operators.until_next import UntilNext
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.until_next import (
    UntilNextNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.utils import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class UntilNextOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_with_timeout(self):
        a = event_set(timestamps=[0, 10, 11, 20, 30])
        b = event_set(timestamps=[1, 12, 21, 22, 42])

        expected_output = event_set(
            timestamps=[1, 12, 12, 21, 35],
            features={
                "until_next": [1, 2, 1, 1, math.nan],
            },
        )

        # Run op
        op = UntilNext(input=a.node(), sampling=b.node(), timeout=5)
        instance = UntilNextNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=a, sampling=b)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_no_sampling(self):
        a = event_set(
            timestamps=[0],
            features={"x": ["a"]},
            indexes=["x"],
        )
        b = event_set(
            timestamps=[0],
            features={"x": ["b"]},
            indexes=["x"],
        )

        expected_output = event_set(
            timestamps=[5],
            features={"x": ["a"], "until_next": [math.nan]},
            indexes=["x"],
        )

        # Run op
        op = UntilNext(input=a.node(), sampling=b.node(), timeout=5)
        instance = UntilNextNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=a, sampling=b)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
