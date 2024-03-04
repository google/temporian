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

from absl.testing import absltest, parameterized

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, i32


class CalendarSecondTest(parameterized.TestCase):
    def test_basic(self):
        timestamps = [
            "1970-01-01 00:00:00",
            "1970-01-01 00:00:01",
            "1970-01-01 00:00:59",
            "2023-05-05 12:30:30",
            "2023-12-12 23:59:59",
        ]
        evset = event_set(timestamps=timestamps)

        expected = event_set(
            timestamps=timestamps,
            features={
                "calendar_second": i32([0, 1, 59, 30, 59]),
            },
            same_sampling_as=evset,
        )

        result = evset.calendar_second()
        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
