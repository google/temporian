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

import datetime

from absl.testing import absltest, parameterized

from temporian.core.data.duration_utils import (
    convert_datetime_to_duration,
    normalize_timestamp,
)


class DurationUtilsTest(parameterized.TestCase):
    @parameterized.parameters(
        # (datetime, expected seconds from unix epoch)
        (datetime.datetime(1970, 1, 1), 0.0),
        (datetime.datetime(1960, 1, 1), -315619200.0),
        (datetime.datetime(1969, 12, 31, 23, 59, 59), -1.0),
        (datetime.datetime(2023, 7, 19, 18, 37, 36), 1689791856.0),
    )
    def test_normalize_timestamp_datetime(self, dtime, expected):
        # Pre-epoch datetimes must not raise OSError (was broken on Windows,
        # see issue #395) and must return the correct negative offset.
        self.assertEqual(normalize_timestamp(dtime), expected)

    @parameterized.parameters(
        (datetime.datetime(1970, 1, 1), 0.0),
        (datetime.datetime(1960, 1, 1), -315619200.0),
        (datetime.datetime(1969, 12, 31, 23, 59, 59), -1.0),
        (datetime.datetime(2023, 7, 19, 18, 37, 36), 1689791856.0),
    )
    def test_convert_datetime_to_duration(self, dtime, expected):
        self.assertEqual(convert_datetime_to_duration(dtime), expected)


if __name__ == "__main__":
    absltest.main()
