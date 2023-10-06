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
from temporian.implementation.numpy.operators.test.utils import (
    assertEqualEventSet,
    testOperatorAndImp,
    SetTimezone,
)


class TickCalendarOperatorTest(absltest.TestCase):
    def test_start_end_00_00(self):
        evset = event_set(
            timestamps=[
                "2020-01-01 00:00",
                "2020-03-01 00:00",
            ],
        )
        node = evset.node()

        # Expected output
        expected_output = event_set(
            timestamps=[
                "2020-01-01 00:00",
                "2020-02-01 00:00",
                "2020-03-01 00:00",
            ],
        )

        # Run op
        op = TickCalendar(
            input=node,
            second=0,
            minute=0,
            hour=0,
            mday=1,
            month="*",
            wday="*",
        )
        instance = TickCalendarNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

        # Check that it's exactly the same with env TZ!=UTC defined
        with SetTimezone():
            instance = TickCalendarNumpyImplementation(op)
            testOperatorAndImp(self, op, instance)
            output = instance.call(input=evset)["output"]

            assertEqualEventSet(self, output, expected_output)

    def test_start_end_offset(self):
        evset = event_set(
            timestamps=[
                "2020-01-01 13:04",
                "2020-03-06 19:35",
            ],
        )
        node = evset.node()

        # Expected output
        expected_output = event_set(
            timestamps=[
                "2020-02-01 00:00",
                "2020-03-01 00:00",
            ],
        )

        # Run op
        op = TickCalendar(
            input=node,
            second=0,
            minute=0,
            hour=0,
            mday=1,
            month="*",
            wday="*",
        )
        instance = TickCalendarNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)

    def test_end_of_month_seconds(self):
        # All seconds at mday=31, should only be valid for months 1, 3, 5

        evset = event_set(
            timestamps=[
                datetime(2020, 1, 1, 0, 0, 0),
                datetime(2020, 6, 1, 0, 0, 0),
            ],
        )
        node = evset.node()

        # Expected output
        def seconds_at_01_01(day, month):
            return [datetime(2020, day, month, 1, 1, sec) for sec in range(60)]

        expected_output = event_set(
            timestamps=seconds_at_01_01(1, 31)
            + seconds_at_01_01(3, 31)
            + seconds_at_01_01(5, 31),
        )

        # Run op
        op = TickCalendar(
            input=node,
            second="*",
            minute=1,
            hour=1,
            mday=31,
            month="*",
            wday="*",
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
            second=0,
            minute="*",
            hour="*",
            mday="*",
            month="*",
            wday="*",
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
            second=0,
            minute=0,
            hour="*",
            wday=6,
            mday="*",
            month="*",
        )
        instance = TickCalendarNumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()
