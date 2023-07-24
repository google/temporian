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

from pathlib import Path

from absl.testing import absltest

import temporian as tp

PUBLIC_API_SYMBOLS = {
    "EventSet",
    "EventSetNode",
    "EventSetOrNode",
    "Schema",
    "duration",
    "run",
    "has_leak",
    "event_set",
    "input_node",
    "plot",
    "compile",
    # IO
    "to_csv",
    "from_csv",
    "to_pandas",
    "from_pandas",
    "to_tensorflow",
    # DTYPES
    "float64",
    "float32",
    "int32",
    "int64",
    "bool_",
    "str_",
    "bytes_",
    # SERIALIZATION
    "save",
    "load",
    "save_graph",
    "load_graph",
    # OPERATORS
    "glue",
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
