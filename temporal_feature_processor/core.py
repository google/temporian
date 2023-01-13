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

"""Core module."""

from absl import logging

from temporal_feature_processor import core_pb2 as pb


def does_nothing() -> None:
  logging.info("Hello world")


def create_toy_processor() -> pb.Processor:
  """Create a toy processor.

  Theoretical result of:
    x = t.read_event(table="sales_records")
    y = t.sma(x["sales"], windows=5)

  Returns:
    Toy processor.
  """

  p = pb.Processor()

  # The INPUT_PLACEHOLDER op injects data in the processor based on some input
  # data provided by the user.
  feed_op = pb.Operator(
      id="op_1",
      operator_def_key="INPUT_PLACEHOLDER",
      attributes=[pb.Operator.Attribute(key="table", str="sales_records")],
      outputs=[pb.Operator.EventArgument(key="data", event_id="event_1")],
  )
  p.operators.append(feed_op)

  # Definition of the feed op results.
  p.samplings.append(pb.Sampling(id="sampling_1"))

  p.events.append(
      pb.Event(
          id="event_1",
          sampling_id="sampling_1",
          feature_ids=["sales"],
      ))

  p.features.append(
      pb.Feature(
          id="sales",
          type=pb.Feature.FLOAT,
          sampling_id="sampling_1",
      ))

  # We apply a SMA on the "price" feature using the same sampling rate as the
  # sales.
  sma = pb.Operator(
      id="op_2",
      operator_def_key="SMA",
      attributes=[],
      inputs=[pb.Operator.EventArgument(key="data", event_id="event_1")],
      outputs=[pb.Operator.EventArgument(key="result", event_id="event_2")],
  )
  p.operators.append(sma)

  # Definition of the SMA op results.

  p.events.append(
      pb.Event(
          id="event_2",
          sampling_id="sampling_1",
          feature_ids=["sma_sales"],
      ))

  p.features.append(
      pb.Feature(
          id="sma_sales",
          type=pb.Feature.FLOAT,
          sampling_id="sampling_1",
      ))

  return p
