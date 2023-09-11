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


from datetime import datetime, timedelta
from absl.testing import absltest

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
    def test_end_of_month_seconds(self):
        # All seconds at mday=31, should only be valid for months 1, 3, 5

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
            min_mday=31,
            max_mday=31,
            min_month=1,
            max_month=12,
            min_wday=0,
            max_wday=6,
        )
        instance = TickCalendarNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_end_of_year_minutes(self):
        # All hours/minutes from 30/12/2019 to 2/1/2020

        evset = event_set(
            timestamps=[
                # 4 days: 2 on 2019 + 2 on 2020
                datetime(2019, 12, 30, 0, 0, 0),
                datetime(2020, 1, 2, 23, 59, 59),  # 2/1 at 23:59:59
            ],
        )
        node = evset.node()

        # Expected timestamps: all hours/minutes in 4 days
        timestamps = []
        for day, month, year in [
            (30, 12, 2019),
            (31, 12, 2019),
            (1, 1, 2020),
            (2, 1, 2020),
        ]:
            for hour in range(24):
                for minute in range(60):
                    timestamps += [datetime(year, month, day, hour, minute, 0)]
        expected_output = event_set(
            timestamps=timestamps,
        )

        # Run op
        op = TickCalendar(
            input=node,
            min_second=0,
            max_second=0,
            min_minute=0,
            max_minute=59,
            min_hour=0,
            max_hour=23,
            min_mday=1,  # any day
            max_mday=31,
            min_month=1,  # any month
            max_month=12,
            min_wday=0,
            max_wday=6,
        )
        instance = TickCalendarNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_weekdays(self):
        # All exact hours from all saturdays in 2023

        evset = event_set(
            timestamps=[
                datetime(2023, 1, 1),
                datetime(2023, 12, 31, 23, 0, 0),
            ],
        )
        node = evset.node()

        # Expected timestamps: all hours/minutes in 4 days
        timestamps = []
        day = datetime(2023, 1, 7)  # First saturday
        one_week = timedelta(days=7)
        while day.year < 2024:
            for hour in range(24):
                timestamps += [day + timedelta(hours=hour)]
            day += one_week
        expected_output = event_set(
            timestamps=timestamps,
        )

        # Run op
        op = TickCalendar(
            input=node,
            min_second=0,
            max_second=0,
            min_minute=0,
            max_minute=0,
            min_hour=0,
            max_hour=23,  # all hours
            min_mday=1,  # any month day
            max_mday=31,
            min_month=1,  # any month
            max_month=12,
            min_wday=6,  # saturday
            max_wday=6,  # saturday
        )
        instance = TickCalendarNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
