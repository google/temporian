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

from absl import logging
from absl.testing import absltest

from temporal_feature_processor.core import processor
from temporal_feature_processor.core.data import dtype
from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.core.data.feature import Feature
from temporal_feature_processor.core.data.sampling import Sampling
from temporal_feature_processor.core.operators import base
from temporal_feature_processor.proto import core_pb2 as pb

# The name of the operator is defined by the number of inputs and outputs.
# For example "OpO1" has 1 input and 2 outputs.


class OpO1(base.Operator):

  @classmethod
  def build_op_definition(cls) -> pb.OperatorDef:
    return pb.OperatorDef(
        key="OpO1",
        place_holder=True,
        outputs=[pb.OperatorDef.Output(key="output")],
    )

  def __init__(self):
    super().__init__()
    sampling = Sampling(index=[])
    self.add_output(
        "output",
        Event(
            features=[
                Feature("f1", dtype.FLOAT, sampling=sampling, creator=self),
                Feature("f2", dtype.FLOAT, sampling=sampling, creator=self),
            ],
            sampling=sampling,
        ),
    )
    self.check()

  def _get_pandas_implementation(self):
    raise NotImplementedError()


class OpI1O1(base.Operator):

  @classmethod
  def build_op_definition(cls) -> pb.OperatorDef:
    return pb.OperatorDef(
        key="OpI1O1",
        inputs=[pb.OperatorDef.Input(key="input")],
        outputs=[pb.OperatorDef.Output(key="output")],
    )

  def __init__(self, data: Event):
    super().__init__()
    self.add_input("input", data)
    self.add_output(
        "output",
        Event(
            features=[
                Feature(
                    "f1", dtype.FLOAT, sampling=data.sampling(), creator=self
                )
            ],
            sampling=data.sampling(),
        ),
    )
    self.check()

  def _get_pandas_implementation(self):
    raise NotImplementedError()


class OpI2O1(base.Operator):

  @classmethod
  def build_op_definition(cls) -> pb.OperatorDef:
    return pb.OperatorDef(
        key="OpI2O1",
        inputs=[
            pb.OperatorDef.Input(key="input_1"),
            pb.OperatorDef.Input(key="input_2"),
        ],
        outputs=[pb.OperatorDef.Output(key="output")],
    )

  def __init__(self, input_1: Event, input_2: Event):
    super().__init__()
    self.add_input("input_1", input_1)
    self.add_input("input_2", input_2)
    self.add_output(
        "output",
        Event(
            features=[
                Feature(
                    "f1", dtype.FLOAT, sampling=input_1.sampling(), creator=self
                )
            ],
            sampling=input_1.sampling(),
        ),
    )
    self.check()

  def _get_pandas_implementation(self):
    raise NotImplementedError()


class OpI1O2(base.Operator):

  @classmethod
  def build_op_definition(cls) -> pb.OperatorDef:
    return pb.OperatorDef(
        key="OpI1O2",
        inputs=[
            pb.OperatorDef.Input(key="input"),
        ],
        outputs=[
            pb.OperatorDef.Output(key="output_1"),
            pb.OperatorDef.Output(key="output_2"),
        ],
    )

  def __init__(self, data: Event):
    super().__init__()
    self.add_input("input", data)
    self.add_output(
        "output_1",
        Event(
            features=[
                Feature(
                    "f1", dtype.FLOAT, sampling=data.sampling(), creator=self
                )
            ],
            sampling=data.sampling(),
        ),
    )
    self.add_output(
        "output_2",
        Event(
            features=[
                Feature(
                    "f1", dtype.FLOAT, sampling=data.sampling(), creator=self
                )
            ],
            sampling=data.sampling(),
        ),
    )
    self.check()

  def _get_pandas_implementation(self):
    raise NotImplementedError()


class ProcessorTest(absltest.TestCase):

  def test_dependency_basic(self):
    o1 = OpO1()
    o2 = OpI1O1(o1.outputs()["output"])
    o3 = OpO1()
    o4 = OpI2O1(o2.outputs()["output"], o3.outputs()["output"])
    o5 = OpI1O2(o4.outputs()["output"])

    p = processor.infer_processor(
        [o1.outputs()["output"], o3.outputs()["output"]],
        [o5.outputs()["output_1"], o4.outputs()["output"]],
    )
    logging.info("Processor:\n%s", p)

    # The two input operators are not listed.
    self.assertLen(p.operators(), 3)

    # The two samplings created by the two input operators.
    self.assertLen(p.samplings(), 2)
    self.assertLen(p.events(), 6)
    self.assertLen(p.features(), 7)

  def test_dependency_missing_input(self):
    o1 = OpO1()
    o2 = OpI1O1(o1.outputs()["output"])

    with self.assertRaisesRegex(
        ValueError,
        "Missing input features.*from placeholder Operator<key:OpO1,",
    ):
      processor.infer_processor([], [o2.outputs()["output"]])


if __name__ == "__main__":
  absltest.main()
