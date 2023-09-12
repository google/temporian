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

from temporian.core.data.node import input_node
from temporian.core.operators.tick_calendar import tick_calendar, TickCalendar


class TickCalendarOperatorTest(absltest.TestCase):
    def setUp(self):
        self._in = input_node([], is_unix_timestamp=True)

    def test_free_seconds_month(self):
        output = tick_calendar(self._in, second="*", minute=1, hour=1, mday=31)
        op = output.creator
        assert isinstance(op, TickCalendar)
        self.assertEqual(op.second, "*")
        self.assertEqual(op.minute, 1)
        self.assertEqual(op.hour, 1)
        self.assertEqual(op.mday, 31)
        self.assertEqual(op.month, "*")
        self.assertEqual(op.wday, "*")

    def test_free_minutes(self):
        output = tick_calendar(self._in, minute="*")
        op = output.creator
        assert isinstance(op, TickCalendar)
        self.assertEqual(op.second, 0)
        self.assertEqual(op.minute, "*")
        self.assertEqual(op.hour, "*")
        self.assertEqual(op.mday, "*")
        self.assertEqual(op.month, "*")
        self.assertEqual(op.wday, "*")

    def test_month_day(self):
        output = tick_calendar(self._in, mday=5)
        op = output.creator
        assert isinstance(op, TickCalendar)
        self.assertEqual(op.second, 0)
        self.assertEqual(op.minute, 0)
        self.assertEqual(op.hour, 0)
        self.assertEqual(op.mday, 5)
        self.assertEqual(op.month, "*")
        self.assertEqual(op.wday, "*")

    def test_month(self):
        output = tick_calendar(self._in, month=8)
        op = output.creator
        assert isinstance(op, TickCalendar)
        self.assertEqual(op.second, 0)
        self.assertEqual(op.minute, 0)
        self.assertEqual(op.hour, 0)
        self.assertEqual(op.mday, 1)
        self.assertEqual(op.month, 8)
        self.assertEqual(op.wday, "*")

    def test_weekdays(self):
        output = tick_calendar(self._in, wday=6)
        op = output.creator
        assert isinstance(op, TickCalendar)
        self.assertEqual(op.second, 0)
        self.assertEqual(op.minute, 0)
        self.assertEqual(op.hour, 0)
        self.assertEqual(op.mday, "*")
        self.assertEqual(op.month, "*")
        self.assertEqual(op.wday, 6)

    def test_weekdays_month(self):
        output = tick_calendar(self._in, wday=6, month=3)
        op = output.creator
        assert isinstance(op, TickCalendar)
        self.assertEqual(op.second, 0)
        self.assertEqual(op.minute, 0)
        self.assertEqual(op.hour, 0)
        self.assertEqual(op.mday, "*")
        self.assertEqual(op.month, 3)
        self.assertEqual(op.wday, 6)

    def test_weekdays_all_hours(self):
        output = tick_calendar(self._in, wday=6, hour="*")
        op = output.creator
        assert isinstance(op, TickCalendar)
        self.assertEqual(op.second, 0)
        self.assertEqual(op.minute, 0)
        self.assertEqual(op.hour, "*")
        self.assertEqual(op.mday, "*")
        self.assertEqual(op.month, "*")
        self.assertEqual(op.wday, 6)


if __name__ == "__main__":
    absltest.main()
