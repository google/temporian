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

  # Create the following processor:
  # x = t.read_event(table="sales_records")
  # y = t.sma(x["sales"], windows=5)

  p = pb.Processor()

  feed_op_id = "op_1"
  feed_output_event_sequence_id = "event_sequence_1"
  feed_output_timestamps_id = "timestamps_1"

  # The INPUT_PLACEHOLDER op injects data in the processor based on some input
  # data provided by the user.
  feed_op = pb.OperatorInstance(
      id=feed_op_id,
      operator_def_key="INPUT_PLACEHOLDER",
      attributes=[
          pb.OperatorInstance.Attribute(key="table", str="sales_records")
      ],
      outputs=[
          pb.OperatorInstance.Output(
              key="data", event_sequence_id=feed_output_event_sequence_id)
      ])
  p.operators.append(feed_op)

  # Definition of the feed op results.
  p.timestamps.append(pb.Timestamps(id=feed_output_timestamps_id))
  p.event_sequences.append(
      pb.EventSequence(
          id=feed_output_event_sequence_id,
          timestamp_id=feed_output_timestamps_id,
          features=[
              pb.EventSequence.Feature(
                  key="price", type=pb.EventSequence.Feature.FLOAT)
          ]))

  # We apply a SMA on the "price" feature using the same sampling rate as the
  # sales.
  sma_op_id = "op_2"
  sma_output_event_sequence_id = "event_sequence_2"

  sma = pb.OperatorInstance(
      id=sma_op_id,
      operator_def_key="SMA",
      inputs=[
          pb.OperatorInstance.Input(
              key="input",
              feature_sequence=pb.OperatorInstance.Input.FeatureSequence(
                  event_sequence_id=feed_output_event_sequence_id,
                  feature_key="price")),
          pb.OperatorInstance.Input(
              key="sampling", timestamp_id=feed_output_timestamps_id)
      ],
      attributes=[],
      outputs=[
          pb.OperatorInstance.Output(
              key="result",
              feature_sequence=pb.OperatorInstance.Output.FeatureSequence(
                  event_sequence_id=sma_output_event_sequence_id,
                  feature_key="sma_price"))
      ])
  p.operators.append(sma)

  # Definition of the SMA op results.
  p.event_sequences.append(
      pb.EventSequence(
          id=sma_output_event_sequence_id,
          # The same timestamps as the SMA input
          timestamp_id=feed_output_timestamps_id,
          features=[
              pb.EventSequence.Feature(
                  key="sma_price", type=pb.EventSequence.Feature.FLOAT)
          ]))

  return p
