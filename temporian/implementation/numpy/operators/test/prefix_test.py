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


import pandas as pd
from absl.testing import absltest

from temporian.core.operators.prefix import Prefix
from temporian.implementation.numpy.operators.prefix import (
    PrefixNumpyImplementation,
)
from temporian.implementation.numpy.data.event_set import EventSet


class PrefixOperatorTest(absltest.TestCase):
    def test_base(self):
        evset = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3],
                    "a": [1.0, 2.0, 3.0],
                    "b": [5, 6, 7],
                    "x": [1, 1, 2],
                }
            ),
            index_names=["x"],
        )
        node = evset.node()

        expected_output = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3],
                    "hello_a": [1.0, 2.0, 3.0],
                    "hello_b": [5, 6, 7],
                    "x": [1, 1, 2],
                }
            ),
            index_names=["x"],
        )

        # Run op
        op = Prefix("hello_", node=node)
        instance = PrefixNumpyImplementation(op)
        output = instance.call(node=evset)["node"]

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
