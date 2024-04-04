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
from absl.testing import absltest
from absl.testing.parameterized import TestCase
from pytz.exceptions import UnknownTimeZoneError

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, i32


class CalendarTimezoneTest(TestCase):
    def setUp(self):
        # Timezone: UTC-3 (America/Montevideo)
        self.tz_offset = -3
        self.tz_name = "America/Montevideo"
        self.tz = timezone(timedelta(hours=self.tz_offset))

        self.timestamps = [
            datetime(2020, 1, 1, 0, 0),  # UTC
            datetime(2020, 1, 1, 2, 0, tzinfo=self.tz),  # UTC-3
        ]
        self.evset = event_set(timestamps=self.timestamps)

    def test_hour(self):
        expected = event_set(
            timestamps=self.timestamps,
            features={"calendar_hour": i32([21, 2])},
            same_sampling_as=self.evset,
        )
        # Check using timezone offset
        assertOperatorResult(
            self, self.evset.calendar_hour(self.tz_offset), expected
        )
        # Check using timezone name
        assertOperatorResult(
            self, self.evset.calendar_hour(self.tz_name), expected
        )

    def test_day_of_month(self):
        # NOTE: 01/01 00:00 in UTC will be 31/12 in UTC-3
        expected = event_set(
            timestamps=self.timestamps,
            features={"calendar_day_of_month": i32([31, 1])},
            same_sampling_as=self.evset,
        )

        # Check using timezone name
        assertOperatorResult(
            self, self.evset.calendar_day_of_month(self.tz_name), expected
        )

    def test_day_of_year(self):
        expected = event_set(
            timestamps=self.timestamps,
            features={"calendar_day_of_year": i32([365, 1])},
            same_sampling_as=self.evset,
        )
        # Check using timezone name
        assertOperatorResult(
            self, self.evset.calendar_day_of_year(self.tz_name), expected
        )

    def test_day_of_week(self):
        expected = event_set(
            timestamps=self.timestamps,
            # 1/1/2020: Wed (day_of_week=2)
            features={"calendar_day_of_week": i32([1, 2])},
            same_sampling_as=self.evset,
        )
        # Check using timezone name
        assertOperatorResult(
            self, self.evset.calendar_day_of_week(self.tz_name), expected
        )

    def test_year(self):
        expected = event_set(
            timestamps=self.timestamps,
            features={"calendar_year": i32([2019, 2020])},
            same_sampling_as=self.evset,
        )
        # Check using timezone name
        assertOperatorResult(
            self, self.evset.calendar_year(self.tz_name), expected
        )

    def test_minute(self):
        # Use fractional timezone offset for this one
        tz_offset = 1.5

        expected = event_set(
            timestamps=self.timestamps,
            features={"calendar_minute": i32([30, 30])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(
            self, self.evset.calendar_minute(tz_offset), expected
        )

    def test_second(self):
        # Use fractional timezone offset for this one
        tz_offset = 36 / 3600  # 36 seconds (in hours)

        expected = event_set(
            timestamps=self.timestamps,
            features={"calendar_second": i32([36, 36])},
            same_sampling_as=self.evset,
        )
        assertOperatorResult(
            self, self.evset.calendar_second(tz_offset), expected
        )

    def test_invalid_timezone(self):
        with self.assertRaises(ValueError):
            self.evset.calendar_hour(tz="I'm a fake timezone")

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            self.evset.calendar_hour(tz=None)


if __name__ == "__main__":
    absltest.main()
