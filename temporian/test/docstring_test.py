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
        tested_modules = set()
        for api_object in tp.__dict__.values():
            if inspect.ismodule(api_object):
                module = api_object
            else:
                module = inspect.getmodule(api_object)
                if module is None:
                    continue

            # Avoid testing twice
            module_name = module.__name__
            if "temporian" not in module_name or module_name in tested_modules:
                continue

            tested_modules.add(module.__name__)
            try:
                # Run with np.pd,tp + module globals as exec context
                _, test_count = doctest.testmod(
                    module,
                    globs={"np": np, "tp": tp, "pd": pd},
                    extraglobs=module.__dict__,
                    raise_on_error=True,
                )
                print(f"Checked {test_count} doc examples on: {module_name}")

            # Failure due to mismatch on expected result
            except doctest.DocTestFailure as e:
                ex = e.example
                path = inspect.getfile(module)
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
                path = inspect.getfile(module)
                raise AssertionError(
                    "Exception running docstring example on file"
                    f" {path}:{ex.lineno}\n>>> {ex.source}{e.exc_info[0]}:"
                    f" {e.exc_info[1]}"
                ) from e


if __name__ == "__main__":
    absltest.main()
