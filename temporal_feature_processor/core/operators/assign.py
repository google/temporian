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

"""Simple moving average operator."""

from temporal_feature_processor.core import operator_lib
from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.core.data.feature import Feature
from temporal_feature_processor.core.operators.base import Operator
from temporal_feature_processor.implementation.pandas.operators.assign import PandasAssignOperator
from temporal_feature_processor.implementation.pandas.operators.base import PandasOperator
from temporal_feature_processor.proto import core_pb2 as pb


class AssignOperator(Operator):
  """Simple moving average operator."""

  def __init__(
      self,
      assignee_event: Event,
      assigned_event: Event,
  ):
    super().__init__()

    # inputs
    self.add_input("assignee_event", assignee_event)
    self.add_input("assigned_event", assigned_event)

    # outputs
    output_features = assignee_event.features() + [
        Feature(name=feature.name(), dtype=feature.dtype(), creator=self)
        for feature in assigned_event.features()
    ]
    output_sampling = assignee_event.sampling()
    self.add_output(
        "output",
        Event(features=output_features, sampling=output_sampling),
    )
    self.check()

  @classmethod
  def build_op_definition(cls) -> pb.OperatorDef:
    return pb.OperatorDef(
        key="ASSIGN",
        inputs=[
            pb.OperatorDef.Input(key="assignee_event"),
            pb.OperatorDef.Input(key="assigned_event"),
        ],
        outputs=[pb.OperatorDef.Output(key="output")],
    )

  def _get_pandas_implementation(self) -> PandasOperator:
    return PandasAssignOperator()


operator_lib.register_operator(AssignOperator)


def assign(
    assignee_event: Event,
    assigned_event: Event,
) -> Event:
  return AssignOperator(assignee_event, assigned_event).outputs()["output"]
