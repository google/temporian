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
from pathlib import Path

import temporian as tp
from temporian.core.operator_lib import registered_operators

PUBLIC_API_SYMBOLS = {
    "EventSet",
    "Node",
    "Schema",
    "duration",
    "run",
    "event_set",
    "input_node",
    "plot",
    "compile",
    # IO
    "to_csv",
    "from_csv",
    "to_pandas",
    "from_pandas",
    # DTYPES
    "float64",
    "float32",
    "int32",
    "int64",
    "bool_",
    "str_",
    # SERIALIZATION
    "load",
    "save",
    "save_graph",
    # OPERATORS
    "cast",
    "drop_index",
    "enumerate",
    "filter",
    "glue",
    "join",
    "add_index",
    "set_index",
    "lag",
    "leak",
    "prefix",
    "propagate",
    "resample",
    "select",
    "rename",
    "since_last",
    "begin",
    "end",
    "tick",
    "timestamps",
    "unique_timestamps",
    # BINARY OPERATORS
    "add",
    "subtract",
    "multiply",
    "divide",
    "floordiv",
    "modulo",
    "power",
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    # CALENDAR OPERATORS
    "calendar_day_of_month",
    "calendar_day_of_week",
    "calendar_day_of_year",
    "calendar_hour",
    "calendar_iso_week",
    "calendar_minute",
    "calendar_month",
    "calendar_second",
    "calendar_year",
    # SCALAR OPERATORS
    "add_scalar",
    "subtract_scalar",
    "multiply_scalar",
    "divide_scalar",
    "floordiv_scalar",
    "modulo_scalar",
    "power_scalar",
    "equal_scalar",
    "not_equal_scalar",
    "greater_equal_scalar",
    "greater_scalar",
    "less_equal_scalar",
    "less_scalar",
    # UNARY OPERATORS
    "abs",
    "log",
    "invert",
    "isnan",
    "notnan",
    # WINDOW OPERATORS
    "simple_moving_average",
    "moving_standard_deviation",
    "cumsum",
    "moving_sum",
    "moving_count",
    "moving_min",
    "moving_max",
}


class PublicAPITest(absltest.TestCase):
    def test_public_symbols(self):
        """Asserts that the symbols exposed under tp.<> are exactly the
        ones we expect."""
        symbols = {s for s in dir(tp) if not s.startswith("__")}
        self.assertEqual(PUBLIC_API_SYMBOLS, symbols)

    def test_public_symbols_in_docs(self):
        """Asserts that all symbols exposed under tp.<> have an auto generated
        documentation page.

        Note that this might need to change in the future, since for example we
        might want to group some symbols under the same page."""
        ref_path = Path("docs/src/reference")
        ref_files = ref_path.rglob("*.md")
        ref_files_names = {file.with_suffix("").name for file in ref_files}
        for symbol in PUBLIC_API_SYMBOLS:
            self.assertIn(symbol, ref_files_names)


if __name__ == "__main__":
    absltest.main()
