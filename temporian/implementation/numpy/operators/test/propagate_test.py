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

from temporian.core.operators.propagate import Propagate
from temporian.implementation.numpy.operators.propagate import (
    PropagateNumpyImplementation,
)
from temporian.implementation.numpy.data.event_set import EventSet


class PropagateOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        evset = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3],
                    "a": [1, 2, 3],
                    "x": [1, 1, 2],
                }
            ),
            index_names=["x"],
        )
        event = evset.node()

        sampling_evset = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 1, 1, 1],
                    "x": [1, 1, 2, 2],
                    "y": [1, 2, 1, 2],
                }
            ),
            index_names=["x", "y"],
        )
        sampling_node = sampling_evset.node()

        expected_output = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 1, 2, 3, 3],
                    "a": [1, 2, 1, 2, 3, 3],
                    "x": [1, 1, 1, 1, 2, 2],
                    "y": [1, 1, 2, 2, 1, 2],
                }
            ),
            index_names=["x", "y"],
        )
        # Run op
        op = Propagate(node=node, sampling=sampling_node)
        instance = PropagateNumpyImplementation(op)
        output = instance.call(node=evset, sampling=sampling_evset)["node"]
        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
