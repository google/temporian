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
"""Utilities for beam unit tests."""

import os
from typing import Optional
import tempfile

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
)
from temporian.beam.io import read_csv, write_csv
from temporian.beam.evaluation import run
from temporian.io.csv import to_csv, from_csv
from temporian.core.data.node import Node
from temporian.implementation.numpy.data.event_set import EventSet


def check_beam_implementation(
    self,
    input_data: EventSet,
    output_node: Node,
    input_node: Optional[Node] = None,
):
    """Checks the result of the Numpy backend against the Beam backend.

    Args:
        input_data: An event set to feed to a graph.
        output_node: Output of the graph.
        input_node: Input of the graph. If not set, uses input_data.node()
            instead.
    """

    if input_node is None:
        input_node = input_data.node()

    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, "input.csv")
    output_path = os.path.join(tmp_dir, "output.csv")

    # Export input data to csv
    to_csv(input_data, path=input_path)

    # Utility to print the intermediate results
    def my_print(x, tag):
        print(f"[{tag}] {x}")
        return x

    # Run the Temporian program using the Beam backend
    with TestPipeline() as p:
        output = (
            p
            | read_csv(input_path, input_node.schema)
            | "Raw input" >> beam.Map(my_print, "input")
            | run(input=input_node, output=output_node)
            | "Raw output" >> beam.Map(my_print, "output")
            | write_csv(output_path, output_node.schema, shard_name_template="")
        )
        assert_that(
            output,
            equal_to([output_path]),
        )

    beam_output = from_csv(
        output_path, index_names=output_node.schema.index_names()
    )

    # Run the Temporian program using the numpy backend
    expected_output = output_node.evaluate(input_data)

    assertEqualEventSet(self, beam_output, expected_output)
