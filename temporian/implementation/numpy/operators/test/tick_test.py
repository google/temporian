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
from temporian.core.operators.tick import Tick
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.tick import (
    TickNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class TickOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_no_rounding(self):
        evset = event_set([1, 5.5])
        expected_output = event_set([1, 5])

        op = Tick(input=evset.node(), interval=4.0, rounding=False)
        instance = TickNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_on_the_spot(self):
        evset = event_set([0, 4, 8])
        expected_output = event_set([0, 4, 8])

        op = Tick(input=evset.node(), interval=4.0, rounding=False)
        instance = TickNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_rounding(self):
        evset = event_set([1, 5.5, 8.1])
        expected_output = event_set([4, 8])

        op = Tick(input=evset.node(), interval=4.0, rounding=True)
        instance = TickNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_empty(self):
        evset = event_set([])
        expected_output = event_set([])

        op = Tick(input=evset.node(), interval=4.0, rounding=False)
        instance = TickNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_no_rounding_single(self):
        evset = event_set([1])
        expected_output = event_set([1])

        op = Tick(input=evset.node(), interval=4.0, rounding=False)
        instance = TickNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_rounding_single(self):
        evset = event_set([1])
        expected_output = event_set([])

        op = Tick(input=evset.node(), interval=4.0, rounding=True)
        instance = TickNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
