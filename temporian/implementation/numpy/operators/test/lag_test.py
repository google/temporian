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

from temporian.core.operators.lag import LagOperator
from temporian.implementation.numpy.operators.lag import LagNumpyImplementation
from temporian.implementation.numpy.data.io import (
    event_set,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class LagNumpyImplementationTest(absltest.TestCase):
    def test_base(self) -> None:
        input_data = event_set(
            timestamps=[1, 2, 3, 4],
            features={"x": [4, 5, 6, 7], "y": [1, 1, 2, 2]},
            index_features=["y"],
        )
        expected_result = event_set(
            timestamps=[1 + 2, 2 + 2, 3 + 2, 4 + 2],
            features={"x": [4, 5, 6, 7], "y": [1, 1, 2, 2]},
            index_features=["y"],
        )
        op = LagOperator(input=input_data.node(), duration=2)
        imp = LagNumpyImplementation(op)
        testOperatorAndImp(self, op, imp)
        filtered_evset = imp.call(input=input_data)["output"]
        assertEqualEventSet(self, filtered_evset, expected_result)


if __name__ == "__main__":
    absltest.main()
