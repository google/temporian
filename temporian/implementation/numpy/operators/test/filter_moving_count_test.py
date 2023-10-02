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

from temporian.core.operators.filter_moving_count import (
    FilterMaxMovingCount,
)
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.filter_moving_count import (
    FilterMaxMovingCountNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.utils import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class FilterMaxMovingCountOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_empty(self):
        evset = event_set([])
        expected_output = event_set([])

        op = FilterMaxMovingCount(input=evset.node(), window_length=1.5)
        instance = FilterMaxMovingCountNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_simple(self):
        evset = event_set([1, 2, 3])
        expected_output = event_set([1, 3])

        op = FilterMaxMovingCount(input=evset.node(), window_length=1.5)
        instance = FilterMaxMovingCountNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_duplicates(self):
        evset = event_set([1, 1, 2, 3])
        expected_output = event_set([1, 3])

        op = FilterMaxMovingCount(input=evset.node(), window_length=1.5)
        instance = FilterMaxMovingCountNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_like_unique(self):
        # TODO: Use "math.nextafter" after drop of python 3.8.
        na = np.nextafter(0, 1)
        evset = event_set([1, 1, na, 2, 3])
        expected_output = event_set([1, na, 2, 3])

        op = FilterMaxMovingCount(input=evset.node(), window_length=na)
        instance = FilterMaxMovingCountNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_like_one_exclusive(self):
        # TODO: Use "math.nextafter" after drop of python 3.8.
        evset = event_set([1, 2, 3])
        expected_output = event_set([1, 2, 3])

        op = FilterMaxMovingCount(input=evset.node(), window_length=1)
        instance = FilterMaxMovingCountNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        assertEqualEventSet(self, output, expected_output)

    def test_like_one_inclusive(self):
        # TODO: Use "math.nextafter" after drop of python 3.8.
        evset = event_set([1, 2, 3])
        expected_output = event_set([1, 3])

        op = FilterMaxMovingCount(
            input=evset.node(), window_length=np.nextafter(1, 2)
        )
        instance = FilterMaxMovingCountNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]
        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
