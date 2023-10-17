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
from temporian.test.utils import SetTimezone, i32


class BaseCalendarTest(TestCase):
    def test_invalid_sampling(self) -> None:
        """
        Test calendar operator with a non-utc timestamp
        sampling.
        """

        with self.assertRaisesRegex(
            ValueError,
            "Calendar operators can only be applied on nodes with unix"
            " timestamps as sampling",
        ):
            event_set(
                timestamps=["1970-01-01 00:00:00"],
                features={"x": [1]},
                indexes=["x"],
                is_unix_timestamp=False,
            ).calendar_day_of_month()

    def test_timezone_defined(self) -> None:
        "Define TZ env var and check that it works identically."

        ts = ["1970-01-01 00:00:00"]
        evset = event_set(timestamps=ts)

        expected = event_set(
            timestamps=ts, features={"calendar_hour": i32([0])}
        )

        result = evset.calendar_hour()

        with SetTimezone():
            tz_result = evset.calendar_hour()

        self.assertEqual(expected, result)
        self.assertEqual(expected, tz_result)


if __name__ == "__main__":
    absltest.main()
