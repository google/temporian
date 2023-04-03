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
import numpy as np
import math
from temporian.core.data.sampling import Sampling
from temporian.core.operators.prefix import Prefix
from temporian.implementation.numpy.operators.prefix import (
    PrefixNumpyImplementation,
)
from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.core.data import event as event_lib
from temporian.core.data import feature as feature_lib
from temporian.core.data import dtype as dtype_lib


class PrefixOperatorTest(absltest.TestCase):
    def test_base(self):
        event_data = NumpyEvent.from_dataframe(
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
        event = event_data.schema()

        expected_output = NumpyEvent.from_dataframe(
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
        op = Prefix("hello_", event=event)
        instance = PrefixNumpyImplementation(op)
        output = instance.call(event=event_data)["event"]

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
