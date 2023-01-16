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

from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.core.data.sampling import Sampling
from temporal_feature_processor.core.operators import base
from temporal_feature_processor.proto import core_pb2 as pb


class OperatorTest(absltest.TestCase):

  def test_check_operator(self):
    class ToyOperator(base.Operator):

      @classmethod
      def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="TOY",
            inputs=[
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="optional_input", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

      def _get_pandas_implementation(self):
        raise NotImplementedError()

    def build_fake_event():
      return Event(features=[], sampling=Sampling(index=[]))

    t = ToyOperator()
    with self.assertRaisesRegex(ValueError, 'Missing input "input"'):
      t.check()

    t.add_input("input", build_fake_event())
    with self.assertRaisesRegex(ValueError, 'Missing output "output"'):
      t.check()

    with self.assertRaisesRegex(ValueError, 'Already existing input "input"'):
      t.add_input("input", build_fake_event())

    t.add_output("output", build_fake_event())
    t.check()

    with self.assertRaisesRegex(ValueError, 'Already existing output "output"'):
      t.add_output("output", build_fake_event())

    t.add_output("unexpected_output", build_fake_event())
    with self.assertRaisesRegex(
        ValueError, 'Unexpected output "unexpected_output"'
    ):
      t.check()


if __name__ == "__main__":
  absltest.main()
