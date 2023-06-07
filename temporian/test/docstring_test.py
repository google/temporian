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
import inspect
from pathlib import Path


from absl.testing import absltest

import numpy as np
import pandas as pd
import temporian as tp

class DocstringsTest(absltest.TestCase):
    """
    Tests all code examples included in docstrings, using builtin doctest module.
    E.g:
    >>> print("hello")
    hello

    """
    def test_docstrings(self):
        for path in Path("temporian").rglob("*.py"):
            try:

                # Run with pd,np,tp already loaded
                doctest.testfile(
                    str(path), module_relative=False, raise_on_error=True,
                    globs={"np": np, "pd": pd, "tp": tp}
                )

            # Failure due to mismatch on expected result
            except doctest.DocTestFailure as e:
                ex = e.example
                # Re-raise as bazel assertion
                self.assertEqual(
                    ex.want,
                    e.got,
                    (
                        f"Docstring example failed on file {path}:{ex.lineno}\n"
                        f">>> {ex.source}"
                    ),
                )

            # Failure due to exception raised during code execution
            except doctest.UnexpectedException as e:
                ex = e.example
                raise AssertionError(
                        f"Exception running docstring example on file {path}:{ex.lineno}\n"
                        f">>> {ex.source}"
                        f"{e.exc_info[0]}: {e.exc_info[1]}"
                ) from e

if __name__ == "__main__":
    absltest.main()
