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

"""Checks that the expected operators are registered."""

from absl.testing import absltest

from temporian.core import operator_lib
from temporian.core.operators import all_operators


class RegisteredOperatorsTest(absltest.TestCase):
    def test_base(self):
        # Note: The operators are stored alphabetically.
        expected_operators = [
            "ABS",
            "ADDITION",
            "ADDITION_SCALAR",
            "AND",
            "CALENDAR_DAY_OF_MONTH",
            "CALENDAR_DAY_OF_WEEK",
            "CALENDAR_ISO_WEEK",
            "CALENDAR_YEAR",
            "CALENDAR_MINUTE",
            "CALENDAR_DAY_OF_YEAR",
            "CALENDAR_MONTH",
            "CALENDAR_HOUR",
            "CALENDAR_SECOND",
            "CAST",
            "DIVISION",
            "DIVISION_SCALAR",
            "EQUAL",
            "EQUAL_SCALAR",
            "GREATER_SCALAR",
            "LESS_SCALAR",
            "FILTER",
            "FLOORDIV",
            "FLOORDIV_SCALAR",
            "GLUE",
            "GREATER",
            "GREATER_EQUAL",
            "GREATER_SCALAR",
            "GREATER_EQUAL_SCALAR",
            "INVERT",
            "IS_NAN",
            "LAG",
            "LESS",
            "LESS_EQUAL",
            "LESS_SCALAR",
            "LESS_EQUAL_SCALAR",
            "LOG",
            "MOVING_COUNT",
            "MOVING_STANDARD_DEVIATION",
            "MOVING_SUM",
            "MODULO",
            "MODULO_SCALAR",
            "MULTIPLICATION",
            "MULTIPLICATION_SCALAR",
            "NOT_NAN",
            "NOT_EQUAL",
            "NOT_EQUAL_SCALAR",
            "OR",
            "POWER",
            "POWER_SCALAR",
            "PREFIX",
            "PROPAGATE",
            "SAMPLE",
            "SELECT",
            "SET_INDEX",
            "SIMPLE_MOVING_AVERAGE",
            "SUBTRACTION",
            "SUBTRACTION_SCALAR",
            "SET_INDEX",
            "RENAME",
            "DROP_INDEX",
            "UNIQUE_TIMESTAMPS",
            "MOVING_MAX",
            "MOVING_MIN",
            "SINCE_LAST",
            "XOR",
        ]

        self.assertSetEqual(
            set(operator_lib.registered_operators().keys()),
            set(expected_operators),
        )


if __name__ == "__main__":
    absltest.main()
