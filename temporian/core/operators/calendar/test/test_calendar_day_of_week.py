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

import pandas as pd
from absl.testing import absltest
from absl.testing.parameterized import TestCase

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, i32


class CalendarDayOfWeekTest(TestCase):
    def test_basic(self):
        timestamps = [
            pd.to_datetime("Monday Mar 13 12:00:00 2023", utc=True),
            pd.to_datetime("Tuesday Mar 14 12:00:00 2023", utc=True),
            pd.to_datetime("Friday Mar 17 00:00:01 2023", utc=True),
            pd.to_datetime("Friday Mar 17 23:59:59 2023", utc=True),
            pd.to_datetime("Sunday Mar 19 23:59:59 2023", utc=True),
        ]
        evset = event_set(timestamps=timestamps)

        expected = event_set(
            timestamps=timestamps,
            features={"calendar_day_of_week": i32([0, 1, 4, 4, 6])},
            same_sampling_as=evset,
        )

        result = evset.calendar_day_of_week()
        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
