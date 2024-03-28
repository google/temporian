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
from typing import List

import numpy as np
import pandas as pd
import polars as pl
from absl.testing import absltest

import temporian as tp


class DocTest(absltest.TestCase):
    """
    Run doctest examples in all python and markdown files
    """

    def test_python_docstrings(self):
        test_paths = list(Path("temporian").rglob("*.py"))
        self._run_doctest_in_files(test_paths)

    def test_markdown_docs(self):
        test_paths = [
            path
            for path in Path("docs").rglob("*.md")
            if "ipynb_checkpoints" not in str(path)
        ]
        self._run_doctest_in_files(test_paths)

    def _run_doctest_in_files(self, file_paths: List[Path]):
        """
        Tests all code examples in file_paths using the builtin doctest module.
        E.g:
        >>> print("hello")
        hello

        """
        with tempfile.TemporaryDirectory() as tmp_dirname:
            tmp_dir = Path(tmp_dirname)
            for path in file_paths:
                print(f"Running doctest examples in {path}")
                try:
                    # Run with pd,np,tp already loaded
                    doctest.testfile(
                        str(path),
                        module_relative=False,
                        raise_on_error=True,
                        globs={
                            "np": np,
                            "pd": pd,
                            "pl": pl,
                            "tp": tp,
                            "tmp_dir": tmp_dir,
                        },
                        # Use ... to match anything and ignore different whitespaces
                        optionflags=doctest.ELLIPSIS
                        | doctest.NORMALIZE_WHITESPACE,
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
                            "Docstring example starting at line"
                            f" {lineno} failed on file {path}\n >>> {ex.source}"
                        ),
                    )

                # Failure due to exception raised during code execution
                except doctest.UnexpectedException as e:
                    test = e.test
                    ex = e.example
                    lineno = (test.lineno or 0) + ex.lineno
                    raise AssertionError(
                        "Exception running docstring example starting at line "
                        f"{lineno} on file {path}\n"
                        f">>> {ex.source}{e.exc_info[0]}: {e.exc_info[1]}"
                    ) from None  # Avoid traceback "During handling..."


if __name__ == "__main__":
    absltest.main()
