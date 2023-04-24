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
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.unique_timestamps import (
    UniqueTimestampsNumpyImplementation,
)


class UniqueTimestampsOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        event_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 2, 2, 3, 3, 3, 4],
                    "a": [1, 2, 3, 4, 5, 6, 7, 8],
                    "c": [1, 1, 1, 1, 1, 2, 2, 2],
                }
            ),
            index_names=["c"],
        )
        event = event_data.schema()

        expected_output = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3, 3, 4],
                    "c": [1, 1, 1, 2, 2],
                }
            ),
            index_names=["c"],
        )

        # Run op
        op = UniqueTimestamps(event=event)
        instance = UniqueTimestampsNumpyImplementation(op)
        output = instance.call(event=event_data)["event"]

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
