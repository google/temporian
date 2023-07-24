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
from datetime import datetime, timezone

from absl.testing import absltest

import numpy as np
from temporian.core.operators.timestamps import Timestamps
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.timestamps import (
    TimestampsNumpyImplementation,
)
from temporian.core.data.duration_utils import convert_timestamps_to_datetimes
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class TimestampsOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        evset = event_set(
            timestamps=[-1, 1, 2, 3, 4, 10],
            features={
                "a": [np.nan, 1.0, 2.0, 3.0, 4.0, np.nan],
                "b": ["A", "A", "B", "B", "C", "C"],
            },
            indexes=["b"],
        )
        node = evset.node()

        expected_output = event_set(
            timestamps=[-1, 1, 2, 3, 4, 10],
            features={
                "timestamps": [-1.0, 1.0, 2.0, 3.0, 4.0, 10.0],
                "b": ["A", "A", "B", "B", "C", "C"],
            },
            indexes=["b"],
        )

        # Run op
        op = Timestamps(input=node)
        instance = TimestampsNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_unix_timestamps(self):
        t0 = 1688156488.0
        timestamps = [t0, t0 + 24 * 3600 * 5, t0 + 0.4]
        dtimes = convert_timestamps_to_datetimes(timestamps)

        evset = event_set(
            timestamps=dtimes,
            features={
                "b": ["A", "A", "B"],
            },
            indexes=["b"],
        )
        node = evset.node()

        expected_output = event_set(
            timestamps=timestamps,
            features={
                "timestamps": timestamps,
                "b": ["A", "A", "B"],
            },
            indexes=["b"],
            is_unix_timestamp=True,
        )

        # Run op
        op = Timestamps(input=node)
        instance = TimestampsNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
