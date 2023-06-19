#!/usr/bin/env python

""" Creates the files for a new wrapper.

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
from temporian.core.data.node import Node, create_node_new_features_new_sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class {capitalized_op}(Operator):
    def __init__(self, input: Node, param: float):
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


def {lower_op}(input: Node, param: float) -> Node:
    """<Text>

    Args:
        input: <Text>
        param: <Text>

    Example:
        <Text>

    Returns:
        <Text>
    """

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

        # create output event set
        output_evset = EventSet(data={{}}, schema=output_schema)

        # fill output event set data
        for index_key, index_data in input.data.items():
            output_evset[index_key] = IndexData(
                [],
                np.array([1], dtype=np.float64),
                schema=output_schema,
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
        "//temporian/implementation/numpy:utils",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

    """
        )

    # Operator implementation test
    with open(
        os.path.join(
            "temporian",
            "implementation",
            "numpy",
            "operators",
            "test",
            lower_op + "_test.py",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"""{license_content()}

from absl.testing import absltest

import numpy as np
from temporian.core.operators.{lower_op} import {capitalized_op}
from temporian.implementation.numpy.data.io import event_set
from temporian.implementation.numpy.operators.{lower_op} import (
    {capitalized_op}NumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)

class {capitalized_op}OperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        evset = event_set(
            timestamps=[1,2,3,4],
            features={{
                    "a": [1.0, 2.0, 3.0, 4.0],
                    "b": [5, 6, 7, 8],
                    "c": ["A", "A", "B", "B"],
            }},
            index_features=["c"],
        )
        node = evset.node()

        expected_output = event_set(
            timestamps=[1, 1],
            features={{
                    "c": ["A", "B"],
            }},
            index_features=["c"],
        )

        # Run op
        op = {capitalized_op}(input=node, param=1.0)
        instance = {capitalized_op}NumpyImplementation(op)
        testOperatorAndImp(self, op, instance)
        output = instance.call(input=evset)["output"]

        assertEqualEventSet(self, output, expected_output)


if __name__ == "__main__":
    absltest.main()

"""
        )

    # Operator implementation test  build
    with open(
        os.path.join(
            "temporian", "implementation", "numpy", "operators", "test", "BUILD"
        ),
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            f"""
py_test(
    name = "{lower_op}_test",
    srcs = ["{lower_op}_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        ":test_util",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:io",
        "//temporian/core/operators:{lower_op}",
        "//temporian/implementation/numpy/operators:{lower_op}",
    ],
)
    """
        )

    print(
        """Don't forget to register the new operators in:
- The imports in the top-level init file temporian/__init__.py
- The imports in temporian/implementation/numpy/operators/__init__.py
- The "test_base" function in temporian/core/test/registered_operators_test.py
- The "test_base" function in temporian/implementation/numpy/test/registered_operators_test.py
- The PUBLIC_API_SYMBOLS set in temporian/test/public_symbols_test.py
- The docs docs/src/reference/path/to/operator.md
- The docs API ref's home page docs/reference/index.md
"""
    )


if __name__ == "__main__":
    app.run(main)
