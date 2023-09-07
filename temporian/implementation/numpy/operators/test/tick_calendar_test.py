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


from datetime import datetime
from absl.testing import absltest

import numpy as np
from temporian.core.data.duration_utils import convert_timestamps_to_datetimes
from temporian.core.operators.tick_calendar import TickCalendar
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.tick_calendar import (
    TickCalendarNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class TickCalendarOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        evset = event_set(
            timestamps=[
                datetime(2020, 1, 1, 0, 0, 0),
                datetime(2020, 6, 1, 0, 0, 0),
            ],
        )
        node = evset.node()

        expected_output = event_set(
            timestamps=[
                datetime(2020, 1, 31, 1, 1, 0),
                datetime(2020, 1, 31, 1, 1, 1),
                datetime(2020, 1, 31, 1, 1, 2),
                datetime(2020, 3, 31, 1, 1, 0),
                datetime(2020, 3, 31, 1, 1, 1),
                datetime(2020, 3, 31, 1, 1, 2),
                datetime(2020, 5, 31, 1, 1, 0),
                datetime(2020, 5, 31, 1, 1, 1),
                datetime(2020, 5, 31, 1, 1, 2),
            ],
        )

        # Run op
        op = TickCalendar(
            input=node,
            min_second=0,
            max_second=2,
            min_minute=1,
            max_minute=1,
            min_hour=1,
            max_hour=1,
            min_day_of_month=31,
            max_day_of_month=31,
            min_month=1,
            max_month=12,
            min_day_of_week=0,
            max_day_of_week=6,
        )
        instance = TickCalendarNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        print("Result:")
        print(
            convert_timestamps_to_datetimes(
                output.get_arbitrary_index_data().timestamps
            )
        )

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
