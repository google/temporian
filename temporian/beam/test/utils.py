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
from typing import Union, List
import tempfile
from absl.testing import absltest

from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
)
from temporian.beam.io.csv import from_csv as beam_from_csv
from temporian.beam.io.csv import to_csv as beam_to_csv
from temporian.beam.evaluation import run_multi_io
from temporian.io.csv import to_csv, from_csv
from temporian.core.data.node import EventSetNode
from temporian.implementation.numpy.data.event_set import EventSet


def check_beam_implementation(
    test: absltest.TestCase,
    input_data: Union[EventSet, List[EventSet]],
    output_node: EventSetNode,
):
    """Checks the result of the Numpy backend against the Beam backend.

    Args:
        test: The absl's test.
        input_data: An event set to feed to a graph.
        output_node: Output of the graph.
        input_node: Input of the graph. If not set, uses input_data.node()
            instead.
    """

    if isinstance(input_data, EventSet):
        input_data = [input_data]

    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "output.csv")
    input_paths = []

    # Export input data to csv
    for input_idx, input_evtset in enumerate(input_data):
        input_path = os.path.join(tmp_dir, f"input_{input_idx}.csv")
        input_paths.append(input_path)
        to_csv(input_evtset, path=input_path)

    # Run the Temporian program using the Beam backend
    with TestPipeline() as p:
        input_pcollection = {}
        for input_path, input_evtset in zip(input_paths, input_data):
            input_pcollection[input_evtset.node()] = p | beam_from_csv(
                input_path, input_evtset.node().schema
            )

        output_pcollection = run_multi_io(
            inputs=input_pcollection, outputs=[output_node]
        )

        assert len(output_pcollection) == 1

        output = output_pcollection[output_node] | beam_to_csv(
            output_path, output_node.schema, shard_name_template=""
        )

        assert_that(
            output,
            equal_to([output_path]),
        )

    beam_output = from_csv(
        output_path, indexes=output_node.schema.index_names()
    )

    # Run the Temporian program using the numpy backend
    expected_output = output_node.run(input_data)

    assertEqualEventSet(test, beam_output, expected_output)
