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
import pandas as pd

from temporian.core.data.sampling import Sampling
from temporian.core.operators.propagate import Propagate
from temporian.implementation.numpy.operators.propagate import (
    PropagateNumpyImplementation,
)
from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.core.data import event as event_lib
from temporian.core.data import feature as feature_lib
from temporian.core.data import dtype as dtype_lib


class PropagateOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        event_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3],
                    "a": [1, 2, 3],
                    "x": [1, 1, 2],
                }
            ),
            index_names=["x"],
        )
        event = event_data.schema()

        sampling_data = NumpyEvent.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [1, 1, 1, 1],
                    "x": [1, 1, 2, 2],
                    "y": [1, 2, 1, 2],
                }
            ),
            index_names=["x", "y"],
        )
        sampling = sampling_data.schema()

        expected_output = NumpyEvent.from_dataframe(
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
        op = Propagate(event=event, sampling=sampling)
        instance = PropagateNumpyImplementation(op)
        output = instance.call(event=event_data, sampling=sampling_data)[
            "event"
        ]

        print(output)

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
