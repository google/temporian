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

from temporian.core.operators.end import EndOperator
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.end import (
    EndNumpyImplementation,
)


class EndOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "a": [5, 6, 7, 8],
                "b": ["A", "A", "B", "B"],
            },
            indexes=["b"],
        )
        node = evset.node()

        expected_output = event_set(
            timestamps=[2, 4],
            features={"b": ["A", "B"]},
            indexes=["b"],
        )

        # Run op
        op = EndOperator(input=node)
        instance = EndNumpyImplementation(op)
        output = instance.call(input=evset)["output"]

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
