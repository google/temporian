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

from datetime import datetime, timedelta, timezone

from absl.testing import absltest, parameterized

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import SetTimezone, assertOperatorResult


class TickCalendarOperatorTest(parameterized.TestCase):
    @parameterized.parameters(
        {
            # Test: specify only mday, all days at 00:00
            "span": ["2020-01-01 00:00", "2020-03-01 00:00"],
            "calendar_args": dict(mday=1),
            "left": None,
            "ticks": [
                "2020-01-01 00:00",
                "2020-02-01 00:00",
                "2020-03-01 00:00",
            ],
            "right": None,
        },
        {
            # Test: mday=31, February should be ignored
            "span": ["2020-01-01 00:00", "2020-03-31 00:00"],
            "calendar_args": dict(mday=31),
            "left": "2019-12-31 00:00",
            "ticks": [
                "2020-01-31 00:00",
                "2020-03-31 00:00",
            ],
            "right": None,
        },
        {
            # Test: offsets in hours at beginning and end
            "span": ["2020-01-01 13:04", "2020-03-06 19:35"],
            "calendar_args": dict(mday=1),
            "left": "2020-01-01 00:00",
            "ticks": [
                "2020-02-01 00:00",
                "2020-03-01 00:00",
            ],
            "right": "2020-04-01 00:00",
        },
        {
            # Test: 3 days in the span, should only consider start/end
            "span": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "calendar_args": dict(mday="*"),
            "left": None,
            "ticks": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "right": None,
        },
        {
            # Test: specific mday/hour and offsets
            "span": ["2020-01-02 13:04", "2020-03-05 19:35"],
            "calendar_args": dict(hour=3, mday=5),
            "left": "2019-12-05 03:00",
            "ticks": [
                "2020-01-05 03:00",
                "2020-02-05 03:00",
                "2020-03-05 03:00",
            ],
            "right": "2020-04-05 03:00",
        },
        {
            # Test: Edge cases for range
            "span": ["2020-01-10 13:00", "2020-03-10 12:59:59.99"],
            "calendar_args": dict(hour=13, mday=10),
            "left": None,
            "ticks": [
                "2020-01-10 13:00",
                "2020-02-10 13:00",
            ],
            "right": "2020-03-10 13:00",
        },
        {
            # Test: Edge cases for range
            "span": ["2020-01-10 12:59:59.99", "2020-03-10 13:00"],
            "calendar_args": dict(hour=13, mday=10),
            "left": "2019-12-10 13:00",
            "ticks": [
                "2020-01-10 13:00",
                "2020-02-10 13:00",
                "2020-03-10 13:00",
            ],
            "right": None,
        },
        {
            # Test: Negative timestamps
            "span": ["1930-07-30 12:59:59.99", "1930-09-30 13:00"],
            "calendar_args": dict(hour=13, mday=30),
            "left": "1930-06-30 13:00",
            "ticks": [
                "1930-07-30 13:00",
                "1930-08-30 13:00",
                "1930-09-30 13:00",
            ],
            "right": None,
        },
    )
    def test_base(self, span, left, ticks, right, calendar_args):
        for include_right in [None, True, False]:
            for include_left in [None, True, False]:
                expected = ticks

                if include_right is None:
                    # inclusive on the right by default
                    if right is not None:
                        expected = expected + [right]
                else:
                    calendar_args["include_right"] = include_right
                    if include_right and right is not None:
                        expected = expected + [right]

                if include_left is not None:
                    calendar_args["include_left"] = include_left
                    if include_left and left is not None:
                        expected = [left] + expected

                evset = event_set(timestamps=span)
                expected_evset = event_set(timestamps=expected)
                result = evset.tick_calendar(**calendar_args)

                # NOTE: check_sampling=False still checks timestamps
                assertOperatorResult(
                    self, result, expected_evset, check_sampling=False
                )

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
            + seconds_at_01_01(5, 31)
            + [
                datetime(2020, 7, 31, 1, 1, 0)
            ]  # inclusive on the right by default,
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

        # inclusive on the right by default
        timestamps.append(datetime(2020, 1, 3, 0, 0, 0))

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
        day = datetime(2023, 1, 7, tzinfo=timezone.utc)  # First saturday
        one_week = timedelta(days=7)
        while day.year < 2024:
            for hour in range(24):
                timestamps += [day + timedelta(hours=hour)]
            day += one_week

        # inclusive on the right by default
        timestamps.append(datetime(2024, 1, 6, 0, 0, 0))

        expected_evset = event_set(
            timestamps=timestamps,
        )

        result = evset.tick_calendar(hour="*", wday=5)
        assertOperatorResult(self, result, expected_evset, check_sampling=False)

    def test_invalid_ranges(self):
        evset = event_set(timestamps=["2020-01-01", "2020-06-01"])
        for kwargs in (
            {"second": -1},
            {"second": 60},
            {"minute": -1},
            {"minute": 60},
            {"hour": -1},
            {"hour": 24},
            {"mday": 0},
            {"mday": 32},
            {"mday": -1},  # may be supported in the future
            {"month": -1},
            {"month": 13},
            {"wday": -1},
            {"wday": 7},
        ):
            with self.assertRaisesRegex(
                ValueError, "Value should be '\*' or integer in range"
            ):
                evset.tick_calendar(**kwargs)

    def test_invalid_types(self):
        evset = event_set(timestamps=["2020-01-01", "2020-06-01"])
        for kwargs in (
            {"second": "1"},
            {"minute": "00"},
            {"hour": "00:00"},
            {"month": "January"},
            {"wday": "Sat"},
        ):
            with self.assertRaisesRegex(ValueError, "Non matching type"):
                evset.tick_calendar(**kwargs)  # type: ignore

    def test_undefined_args(self):
        evset = event_set(timestamps=["2020-01-01", "2020-06-01"])
        with self.assertRaisesRegex(
            ValueError,
            "Can't set argument to None because previous and following",
        ):
            evset.tick_calendar(second=1, hour=1)  # undefined min

        with self.assertRaisesRegex(
            ValueError,
            "Can't set argument to None because previous and following",
        ):
            evset.tick_calendar(second=1, month=1)

        with self.assertRaisesRegex(
            ValueError,
            "Can't set argument to None because previous and following",
        ):
            evset.tick_calendar(hour=0, month=1)


if __name__ == "__main__":
    absltest.main()
