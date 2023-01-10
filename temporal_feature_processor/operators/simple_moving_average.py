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


from typing import Optional

from temporal_feature_processor import core_pb2 as pb
from temporal_feature_processor import event as event_lib
from temporal_feature_processor import feature as feature_lib
from temporal_feature_processor import operator
from temporal_feature_processor import operator_lib


class SimpleMovingAverage(operator.Operator):
  """Simple moving average operator."""

  def __init__(
      self,
      data: event_lib.Event,
      window_length: int,
      sampling: Optional[event_lib.Event] = None,
  ):
    super().__init__()
    if sampling is not None:
      self.add_input("sampling", sampling)
    else:
      sampling = data.sampling()

    self.add_input("data", data)

    features = [  # pylint: disable=g-complex-comprehension
        feature_lib.Feature(
            name=f.name(),
            dtype=f.dtype(),
            sampling=sampling,
        )
        for f in data.features()
    ]

    self.add_output(
        "output",
        event_lib.Event(
            features=features,
            sampling=sampling,
        ),
    )

    self.check()

  @classmethod
  def build_op_definition(cls) -> pb.OperatorDef:
    return pb.OperatorDef(
        key="SIMPLE_MOVING_AVERAGE",
        inputs=[
            pb.OperatorDef.Input(key="data"),
            pb.OperatorDef.Input(key="sampling", is_optional=True),
        ],
        outputs=[pb.OperatorDef.Output(key="output")],
    )


operator_lib.register_operator(SimpleMovingAverage)


def sma(
    data: event_lib.Event,
    window_length: int,
    sampling: Optional[event_lib.Event] = None,
) -> event_lib.Event:
  return SimpleMovingAverage(
      data=data,
      window_length=window_length,
      sampling=sampling,
  ).outputs()["output"]
