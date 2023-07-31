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
import doctest
import tempfile
from pathlib import Path

from absl.testing import absltest

import numpy as np
import pandas as pd
import temporian as tp


class MarkdownCodeExamplesTest(absltest.TestCase):
    """
    Tests all code examples in markdown files, using builtin doctest module.
    E.g:
    >>> print("hello")
    hello

    """

    def test_markdown_code_examples(self):
        tmp_dir_handle = tempfile.TemporaryDirectory()
        tmp_dir = Path(tmp_dir_handle.name)
        for path in Path("docs").rglob("*.md"):
            print(f"Testing code examples in {path}")
            try:
                # Run with pd,np,tp already loaded
                doctest.testfile(
                    str(path),
                    module_relative=False,
                    raise_on_error=True,
                    globs={"np": np, "pd": pd, "tp": tp, "tmp_dir": tmp_dir},
                    # Use ... to match anything and ignore different whitespaces
                    optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
                    verbose=False,
                )

            # Failure due to mismatch on expected result
            except doctest.DocTestFailure as e:
                test = e.test
                ex = e.example
                lineno = (test.lineno or 0) + ex.lineno
                # Re-raise as bazel assertion
                self.assertEqual(
                    ex.want,
                    e.got,
                    (
                        f"Docstring example starting at line {lineno} failed on"
                        f" file {path}\n >>> {ex.source}"
                    ),
                )

            # Failure due to exception raised during code execution
            except doctest.UnexpectedException as e:
                test = e.test
                ex = e.example
                lineno = (test.lineno or 0) + ex.lineno
                # pylint: disable=raise-missing-from
                raise AssertionError(
                    "Exception running docstring example starting at line "
                    f"{lineno} on file {path}\n"
                    f">>> {ex.source}{e.exc_info[0]}: {e.exc_info[1]}"
                )
                # pylint: enable=raise-missing-from
        tmp_dir_handle.cleanup()


if __name__ == "__main__":
    absltest.main()
