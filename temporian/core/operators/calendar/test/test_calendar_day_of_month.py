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


class CalendarDayOfMonthTest(TestCase):
    def test_no_index(self):
        timestamps = [
            "1970-01-01 00:00:00",
            "2023-03-14 00:00:01",
            "2023-03-14 23:59:59",
            "2023-03-15 12:00:00",
        ]
        evset = event_set(timestamps=timestamps)

        expected = event_set(
            timestamps=timestamps,
            features={"calendar_day_of_month": i32([1, 14, 14, 15])},
            same_sampling_as=evset,
        )

        result = evset.calendar_day_of_month()
        assertOperatorResult(self, result, expected)

    def test_with_index(self):
        timestamps = [
            "1970-01-01 00:00:00",
            "2023-03-14 00:00:01",
            "2023-03-14 00:00:01",
            "2023-03-14 23:59:59",
            "2023-03-15 12:00:00",
        ]
        evset = event_set(
            timestamps=timestamps,
            features={"x": [1, 1, 2, 1, 2]},
            indexes=["x"],
            is_unix_timestamp=True,
        )

        expected = event_set(
            timestamps=timestamps,
            features={
                "x": [1, 1, 2, 1, 2],
                "calendar_day_of_month": i32([1, 14, 14, 14, 15]),
            },
            indexes=["x"],
            same_sampling_as=evset,
            is_unix_timestamp=True,
        )

        result = evset.calendar_day_of_month()
        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
