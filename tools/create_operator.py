#!/usr/bin/env python

"""Creates the files for a new wrapper.

The changes are:
    - Create operator definition + add to build file
    - Create operator implementation + add to build file
    - Create operator implementation test + add to build file

Usage example:
    From temporian root directory i.e., the directory containing README.md.
    ./tools/create_operator.py --operator=rename

"""

import os
import string

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "operator",
    "",
    "Name of the operator. Should be lower-case with _ e.g., my_new_operator",
)


def license_content():
    """Google license."""

    return """# Copyright 2021 Google LLC.
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
"""


def main(argv):
    del argv
    if not FLAGS.operator:
        raise ValueError("--operator not set")

    # Validate input
    raw_op = FLAGS.operator
    allowed = list(string.ascii_lowercase) + ["_"]
    for c in raw_op:
        if c not in allowed:
            raise ValueError(f"Character '{c}' not allowed")

    # Example:
    # raw_op = lower_op = "hello_world"
    # upper_op = "HELLO_WOLRD"
    # capitalized_op = "HelloWorld"
    lower_op = raw_op.lower()
    upper_op = lower_op.upper()
    capitalized_op = upper_op.replace("_", " ").title().replace(" ", "")

    # Operator
    with open(
        os.path.join("temporian", "core", "operators", lower_op + ".py"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f'''{license_content()}

"""{capitalized_op} operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import EventSetNode, create_node_new_features_new_sampling
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck


class {capitalized_op}(Operator):
    def __init__(self, input: EventSetNode, param: float):
        super().__init__()

        self.add_input("input", input)
        self.add_attribute("param", param)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=[],
                indexes=input.schema.indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="{upper_op}",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="param",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator({capitalized_op})


@typecheck
@compile
def {lower_op}(input: EventSetOrNode, param: float) -> EventSetOrNode:
    """<Text>

    Example:

        ```python
        >>> a = tp.event_set(timestamps=[0, 1, 2], features={{"A": [0, 10, 20]}})
        >>> b = tp.{lower_op}(a)
        >>> b
        indexes: []
        features: [('A', int64)]
        events:
            (3 events):
                timestamps: [0. 1. 2.]
                'A': [ 0 10 20]
        ...

        ```

    Args:
        input: <Text>
        param: <Text>

    Returns:
        <Text>
    """
    assert isinstance(input, EventSetNode)

    return {capitalized_op}(input=input, param=param).outputs["output"]

'''
        )

    # Operator build
    with open(
        os.path.join("temporian", "core", "operators", "BUILD"),
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            f"""
py_library(
    name = "{lower_op}",
    srcs = ["{lower_op}.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/proto:core_py_proto",
    ],
)
    """
        )

    # Operator implementation
    with open(
        os.path.join(
            "temporian",
            "implementation",
            "numpy",
            "operators",
            lower_op + ".py",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f'''{license_content()}

"""Implementation for the {capitalized_op} operator."""


from typing import Dict
import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.{lower_op} import {capitalized_op}
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation

class {capitalized_op}NumpyImplementation(OperatorImplementation):

    def __init__(self, operator: {capitalized_op}) -> None:
        assert isinstance(operator, {capitalized_op})
        super().__init__(operator)

    def __call__(
        self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, {capitalized_op})

        output_schema = self.output_schema("output")

        # Create output EventSet
        output_evset = EventSet(data={{}}, schema=output_schema)

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=[],
                    timestamps=np.array([1], dtype=np.float64),
                    schema=output_schema,
                )
            )

        return {{"output": output_evset}}


implementation_lib.register_operator_implementation(
    {capitalized_op}, {capitalized_op}NumpyImplementation
)
'''
        )

    # Operator implementation build
    with open(
        os.path.join(
            "temporian", "implementation", "numpy", "operators", "BUILD"
        ),
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            f"""
py_library(
    name = "{lower_op}",
    srcs = ["{lower_op}.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        ":base",
        "//temporian/core/data:duration_utils",
        "//temporian/core/operators:{lower_op}",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

    """
        )

    # Operator implementation test
    with open(
        os.path.join(
            "temporian", "core", "operators", "test", f"test_{lower_op}.py"
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"""{license_content()}

from absl.testing import absltest, parameterized

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult, f64, i32


class {capitalized_op}Test(parameterized.TestCase):
    def test_empty(self):
        evset = event_set(timestamps=[], features={{"a": i32([])}})
        result = evset.{lower_op}()
        expected = event_set(timestamps=[], same_sampling_as=evset)

        assertOperatorResult(self, result, expected, check_sampling=True)

    @parameterized.parameters(
        {{"in_timestamps": [0.0, 1.0], "out_timestamps": [0.0, 1.0]}},
        {{"in_timestmaps": [1.0, 2.0], "out_timestamps": [1.0, 2.0]}},
    )
    def test_base(self, in_timestamps, out_timestamps):
        evset = event_set(
            timestamps=in_timestamps,
            features={{
                "a": f64([0, 0]),
            }},
        )
        result = evset.{lower_op}()
        expected = event_set(
            timestamps=out_timestamps,
            features={{
                "a": f64([0, 0]),
            }},
        )

        assertOperatorResult(self, result, expected, check_sampling=False)


if __name__ == "__main__":
    absltest.main()

"""
        )

    # Operator implementation test  build
    with open(
        os.path.join("temporian", "core", "operators", "test", "BUILD"),
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            f"""
py_test(
    name = "test_{lower_op}",
    srcs = ["test_{lower_op}.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:io",
        # "//temporian/core/data:duration",
        "//temporian/test:utils",
    ],
)
    """
        )

    print(
        """\
Don't forget to update the following code:
 1. The imports in the top-level init file temporian/__init__.py (if global)
 2. The EventSetOperations class in temporian/core/event_set_ops.py (if not global)
 3. Move the docstring from the operator's .py file to the EventSetOperations class (if not global)
 4. The imports in temporian/implementation/numpy/operators/__init__.py
 5. The "operators" py_library in temporian/implementation/numpy/operators/BUILD
 6. The "test_base" function in temporian/core/test/registered_operators_test.py
 7. The "test_base" function in temporian/implementation/numpy/test/registered_operators_test.py
 8. The PUBLIC_API_SYMBOLS set in temporian/test/public_api_test.py (if global)
 9. The .md file in docs/src/reference/temporian/operators
10. The docs API ref's home page docs/src/reference/index.md
11. Write unit tests in temporian/core/operators/test
12. Once your op is implemented, run `python tools/build_cleaner.py` and fix Bazel dependencies
"""
    )


if __name__ == "__main__":
    app.run(main)
