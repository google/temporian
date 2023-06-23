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

import pandas as pd
from temporian.core.operators.unique_timestamps import UniqueTimestamps
from temporian.implementation.numpy.operators.unique_timestamps import (
    UniqueTimestampsNumpyImplementation,
)
from temporian.io.pandas import from_pandas


class UniqueTimestampsOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        evset = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 2, 2, 3, 3, 3, 4],
                    "a": [1, 2, 3, 4, 5, 6, 7, 8],
                    "c": [1, 1, 1, 1, 1, 2, 2, 2],
                }
            ),
            indexes=["c"],
        )
        node = evset.node()

        expected_output = from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3, 3, 4],
                    "c": [1, 1, 1, 2, 2],
                }
            ),
            indexes=["c"],
        )

        # Run op
        op = UniqueTimestamps(input=node)
        instance = UniqueTimestampsNumpyImplementation(op)
        output = instance.call(input=evset)["output"]

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
