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
from temporian.implementation.numpy import operators as imps
from temporian.implementation.numpy import implementation_lib


class RegisteredOperatorsTest(absltest.TestCase):
    def test_base(self):
        # Note: The operators are stored alphabetically.
        expected_implementations = [
            "ABS",
            "ADDITION",
            "ADDITION_SCALAR",
            "ADD_INDEX",
            "AND",
            "ARCCOS",
            "ARCSIN",
            "ARCTAN",
            "BEGIN",
            "BITWISE_AND",
            "BITWISE_OR",
            "BITWISE_XOR",
            "CALENDAR_DAY_OF_MONTH",
            "CALENDAR_DAY_OF_WEEK",
            "CALENDAR_DAY_OF_YEAR",
            "CALENDAR_HOUR",
            "CALENDAR_ISO_WEEK",
            "CALENDAR_MINUTE",
            "CALENDAR_MONTH",
            "CALENDAR_SECOND",
            "CALENDAR_YEAR",
            "CAST",
            "COMBINE",
            "COS",
            "DIVISION",
            "DIVISION_SCALAR",
            "DROP_INDEX",
            "END",
            "ENUMERATE",
            "EQUAL",
            "EQUAL_SCALAR",
            "FFT",
            "FILTER",
            "FILTER_MAX_MOVING_COUNT",
            "FLOORDIV",
            "FLOORDIV_SCALAR",
            "GLUE",
            "GREATER",
            "GREATER_EQUAL",
            "GREATER_EQUAL_SCALAR",
            "GREATER_SCALAR",
            "INVERT",
            "IS_NAN",
            "JOIN",
            "LAG",
            "LEAK",
            "LESS",
            "LESS_EQUAL",
            "LESS_EQUAL_SCALAR",
            "LESS_SCALAR",
            "LOG",
            "MAP",
            "MODULO",
            "MODULO_SCALAR",
            "MOVING_COUNT",
            "MOVING_MAX",
            "MOVING_MIN",
            "MOVING_STANDARD_DEVIATION",
            "MOVING_SUM",
            "MULTIPLICATION",
            "MULTIPLICATION_SCALAR",
            "NOT_EQUAL",
            "NOT_EQUAL_SCALAR",
            "NOT_NAN",
            "OR",
            "POWER",
            "POWER_SCALAR",
            "PREFIX",
            "PROPAGATE",
            "RENAME",
            "RESAMPLE",
            "SELECT",
            "SELECT_INDEX_VALUES",
            "SIMPLE_MOVING_AVERAGE",
            "SIN",
            "SINCE_LAST",
            "SUBTRACTION",
            "SUBTRACTION_SCALAR",
            "TAN",
            "TICK",
            "TICK_CALENDAR",
            "TIMESTAMPS",
            "UNIQUE_TIMESTAMPS",
            "UNTIL_NEXT",
            "WHERE",
            "XOR",
        ]

        self.assertEqual(
            expected_implementations,
            sorted(list(set(expected_implementations))),
        )

        self.assertSetEqual(
            set(implementation_lib.registered_implementations().keys()),
            set(expected_implementations),
        )


if __name__ == "__main__":
    absltest.main()
