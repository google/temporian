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

from temporian.core.data.node import input_node
from temporian.core.operators import base
from temporian.proto import core_pb2 as pb


class OperatorTest(absltest.TestCase):
    def test_check_operator(self):
        class ToyOperator(base.Operator):
            @classmethod
            def build_op_definition(cls) -> pb.OperatorDef:
                return pb.OperatorDef(
                    key="TOY",
                    inputs=[
                        pb.OperatorDef.Input(key="input"),
                        pb.OperatorDef.Input(
                            key="optional_input", is_optional=True
                        ),
                    ],
                    outputs=[pb.OperatorDef.Output(key="output")],
                )

            def _get_pandas_implementation(self):
                raise NotImplementedError()

        def build_fake_node():
            return input_node(features=[])

        t = ToyOperator()
        with self.assertRaisesRegex(ValueError, 'Missing input "input"'):
            t.check()

        t.add_input("input", build_fake_node())
        with self.assertRaisesRegex(ValueError, 'Missing output "output"'):
            t.check()

        with self.assertRaisesRegex(
            ValueError, 'Already existing input "input"'
        ):
            t.add_input("input", build_fake_node())

        t.add_output("output", build_fake_node())
        t.check()

        with self.assertRaisesRegex(
            ValueError, 'Already existing output "output"'
        ):
            t.add_output("output", build_fake_node())

        t.add_output("unexpected_output", build_fake_node())
        with self.assertRaisesRegex(
            ValueError, 'Unexpected output "unexpected_output"'
        ):
            t.check()

    def test_check_operator_with_key_prefix(self):
        class ToyOperator(base.Operator):
            @classmethod
            def build_op_definition(cls) -> pb.OperatorDef:
                return pb.OperatorDef(
                    key="TOY",
                    inputs=[pb.OperatorDef.Input(key_prefix="input_")],
                )

            def _get_pandas_implementation(self):
                raise NotImplementedError()

        def build_fake_node():
            return input_node(features=[])

        t = ToyOperator()
        t.check()

        t.add_input("input", build_fake_node())
        with self.assertRaisesRegex(ValueError, 'Unexpected input "input"'):
            t.check()

        t = ToyOperator()
        t.add_input("input_1", build_fake_node())
        t.add_input("input_2", build_fake_node())
        t.add_input("input_3", build_fake_node())

        with self.assertRaisesRegex(
            ValueError, 'Already existing input "input_3"'
        ):
            t.add_input("input_3", build_fake_node())

        t.check()

    def test_check_operator_with_key_prefix_and_invalid_def(self):
        class ToyOperator(base.Operator):
            @classmethod
            def build_op_definition(cls) -> pb.OperatorDef:
                return pb.OperatorDef(
                    key="TOY",
                    inputs=[
                        pb.OperatorDef.Input(key_prefix="input_"),
                        pb.OperatorDef.Input(key_prefix="input"),
                    ],
                )

            def _get_pandas_implementation(self):
                raise NotImplementedError()

        def build_fake_node():
            return input_node(features=[])

        t = ToyOperator()
        t.add_input("input_1", build_fake_node())
        with self.assertRaisesRegex(
            ValueError, 'Input "input_1" matches multiple prefix inputs'
        ):
            t.check()


if __name__ == "__main__":
    absltest.main()
