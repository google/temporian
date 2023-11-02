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
from absl.testing import absltest, parameterized

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, SetTimezone


class TickCalendarOperatorTest(parameterized.TestCase):
    @parameterized.parameters(
        {
            # Test: specify only mday, all days at 00:00
            "span": ["2020-01-01 00:00", "2020-03-01 00:00"],
            "calendar_args": dict(mday=1),
            "expected": [
                "2020-01-01 00:00",
                "2020-02-01 00:00",
                "2020-03-01 00:00",
            ],
        },
        {
            # Test: mday=31, February should be ignored
            "span": ["2020-01-01 00:00", "2020-03-31 00:00"],
            "calendar_args": dict(mday=31),
            "expected": [
                "2020-01-31 00:00",
                "2020-03-31 00:00",
            ],
        },
        {
            # Test: offsets in hours at beginning and end
            "span": ["2020-01-01 13:04", "2020-03-06 19:35"],
            "calendar_args": dict(mday=1),
            "expected": ["2020-02-01 00:00", "2020-03-01 00:00"],
        },
        {
            # Test: 3 days in the span, should only consider start/end
            "span": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "calendar_args": dict(mday="*"),
            "expected": ["2020-01-01", "2020-01-02", "2020-01-03"],
        },
        {
            # Test: specific mday/hour and offsets
            "span": ["2020-01-02 13:04", "2020-03-05 19:35"],
            "calendar_args": dict(hour=3, mday=5),
            "expected": ["2020-02-05 03:00", "2020-03-05 03:00"],
        },
        {
            # Test: Edge cases for range
            "span": ["2020-01-10 13:00", "2020-03-10 12:59:59.99"],
            "calendar_args": dict(hour=13, mday=10),
            "expected": ["2020-01-10 13:00", "2020-02-10 13:00"],
        },
    )
    def test_base(self, span, expected, calendar_args):
        evset = event_set(timestamps=span)
        expected_evset = event_set(timestamps=expected)
        result = evset.tick_calendar(**calendar_args)

        # NOTE: check_sampling=False still checks timestamps
        assertOperatorResult(self, result, expected_evset, check_sampling=False)

        # Check that it's exactly the same with env TZ!=UTC defined
        with SetTimezone():
            result = evset.tick_calendar(**calendar_args)
            assertOperatorResult(
                self, result, expected_evset, check_sampling=False
            )

    def test_end_of_month_seconds(self):
        # All seconds at mday=31, should only be valid for months 1, 3, 5

        evset = event_set(timestamps=["2020-01-01", "2020-06-01"])

        # Expected output
        def seconds_at_01_01(day, month):
            return [datetime(2020, day, month, 1, 1, sec) for sec in range(60)]

        expected_evset = event_set(
            timestamps=seconds_at_01_01(1, 31)
            + seconds_at_01_01(3, 31)
            + seconds_at_01_01(5, 31),
        )
        result = evset.tick_calendar(second="*", minute=1, hour=1, mday=31)
        assertOperatorResult(self, result, expected_evset, check_sampling=False)

    def test_end_of_year_minutes(self):
        # All hours/minutes from 30/12/2019 to 2/1/2020
        evset = event_set(
            timestamps=[
                "2019-12-30 00:00:00",
                "2020-01-02 23:59:59",
            ],
        )

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
        expected_evset = event_set(
            timestamps=timestamps,
        )
        result = evset.tick_calendar(minute="*")
        assertOperatorResult(self, result, expected_evset, check_sampling=False)

    def test_weekdays(self):
        # All exact hours from all saturdays in 2023

        evset = event_set(
            timestamps=[
                "2023-01-01",
                "2023-12-31 23:00:00",
            ],
        )

        # Expected timestamps: all hours on Saturdays
        timestamps = []
        day = datetime(2023, 1, 7)  # First saturday
        one_week = timedelta(days=7)
        while day.year < 2024:
            for hour in range(24):
                timestamps += [day + timedelta(hours=hour)]
            day += one_week
        expected_evset = event_set(
            timestamps=timestamps,
        )

        result = evset.tick_calendar(hour="*", wday=5)
        assertOperatorResult(self, result, expected_evset, check_sampling=False)


if __name__ == "__main__":
    absltest.main()
