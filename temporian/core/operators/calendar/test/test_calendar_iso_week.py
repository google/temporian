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
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, i32


class CalendarIsoWeekTest(TestCase):
    def test_basic(self):
        timestamps = [
            "1970-01-01",  # Thursday, so ISO week 1 goes until Sunday 4th
            "1970-01-04",  # Sunday, still ISO week 1
            "1970-01-05",  # Monday, ISO week 2
            "2023-01-01",  # Sunday, ISO week 52 (week 1 is the first to contain a Thursday)
            "2023-01-08",  # Sunday, ISO week 1
            "2023-01-09",  # Monday, ISO week "2023-03-24", ISO week 1"2023-12-31", ISO week 52
            "2023-03-24",  # ISO week 12
            "2023-12-31",  # ISO week 52
        ]
        evset = event_set(timestamps=timestamps)

        expected = event_set(
            timestamps=timestamps,
            features={
                "calendar_iso_week": i32([1, 1, 2, 52, 1, 2, 12, 52]),
            },
            same_sampling_as=evset,
        )

        result = evset.calendar_iso_week()
        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
