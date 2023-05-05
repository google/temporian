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
from temporian.implementation.numpy.operators import all_operators as imps
from temporian.implementation.numpy import implementation_lib


class RegisteredOperatorsTest(absltest.TestCase):
    def test_base(self):
        # Note: The operators are stored alphabetically.
        expected_implementations = [
            "ABS",
            "ADDITION",
            "ADDITION_SCALAR",
            "CALENDAR_DAY_OF_MONTH",
            "CALENDAR_DAY_OF_WEEK",
            "CALENDAR_ISO_WEEK",
            "CALENDAR_YEAR",
            "CALENDAR_DAY_OF_YEAR",
            "CALENDAR_SECOND",
            "CALENDAR_MINUTE",
            "CALENDAR_MONTH",
            "CALENDAR_HOUR",
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
            "MOVING_MIN",
            "MOVING_MAX",
            "MOVING_STANDARD_DEVIATION",
            "MOVING_SUM",
            "MODULO",
            "MODULO_SCALAR",
            "MULTIPLICATION",
            "MULTIPLICATION_SCALAR",
            "NOT_NAN",
            "NOT_EQUAL",
            "NOT_EQUAL_SCALAR",
            "POWER",
            "POWER_SCALAR",
            "PREFIX",
            "PROPAGATE",
            "RENAME",
            "SAMPLE",
            "SELECT",
            "SIMPLE_MOVING_AVERAGE",
            "SUBTRACTION",
            "SUBTRACTION_SCALAR",
            "SET_INDEX",
            "DROP_INDEX",
            "UNIQUE_TIMESTAMPS",
            "SINCE_LAST",
        ]

        self.assertSetEqual(
            set(implementation_lib.registered_implementations().keys()),
            set(expected_implementations),
        )


if __name__ == "__main__":
    absltest.main()
