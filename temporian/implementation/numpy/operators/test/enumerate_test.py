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
from temporian.core.operators.enumerate import Enumerate
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.enumerate import (
    EnumerateNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class EnumerateOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4, 0, 1],
            features={
                "a": [1.0, 2.0, 3.0, 4.0, 0.0, 1.0],
                "b": [5, 6, 7, 8, 1, 2],
                "c": ["A", "A", "A", "A", "B", "B"],
            },
            indexes=["c"],
        )
        node = evset.node()

        expected_output = event_set(
            timestamps=[1, 2, 3, 4, 0, 1],
            features={
                "enum_res": [0, 1, 2, 3, 0, 1],
                "c": ["A", "A", "A", "A", "B", "B"],
            },
            indexes=["c"],
        )

        # Run op
        op = Enumerate(input=node, name="enum_res")
        instance = EnumerateNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
