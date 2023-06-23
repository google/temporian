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

import os
import tempfile

from absl.testing import absltest
from absl import flags
import temporian as tp

import temporian.beam as tpb
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to


def test_data() -> str:
    return os.path.join(flags.FLAGS.test_srcdir, "temporian")


class TFPTest(absltest.TestCase):
    def test_run(self):
        tmp_dir = tempfile.mkdtemp()
        input_path = os.path.join(tmp_dir, "input.csv")
        output_path = os.path.join(tmp_dir, "output.csv")

        # Create a toy dataset and save it in a csv file.
        input_data = tp.event_set(
            timestamps=[1, 2, 13, 14, 15],
            features={
                "a": ["x", "y", "z", "x", "y"],
                "b": [1, 2, 3, 2, 1],
            },
        )
        tp.to_csv(input_data, path=input_path)

        # Create a graph.
        input_node = tp.input_node([("a", tp.str_), ("b", tp.float32)])
        output_node = tp.moving_sum(input_node["b"], 4)

        # Execute the graph in Beam and export the result in a csv file.
        with TestPipeline() as p:
            output = (
                p
                | tpb.read_csv(input_path, input_node.schema)
                | tpb.run(input=input_node, output=output_node)
                | tpb.write_csv(
                    output_path, output_node.schema, shard_name_template=""
                )
            )
            p.run()
            assert_that(output, equal_to([output_path]))

        with open(output_path, "r", encoding="utf-8") as f:
            print("Results:\n" + f.read(), flush=True)

    def test_run_multi_io(self):
        tmp_dir = tempfile.mkdtemp()
        input_path_1 = os.path.join(tmp_dir, "input_1.csv")
        input_path_2 = os.path.join(tmp_dir, "input_2.csv")
        output_path_1 = os.path.join(tmp_dir, "output_1.csv")
        output_path_2 = os.path.join(tmp_dir, "output_2.csv")

        # Create a toy dataset and save it in a csv file.
        input_data_1 = tp.event_set(
            timestamps=[1, 2, 13, 14, 15],
            features={"a": [1, 2, 3, 2, 1]},
        )
        tp.to_csv(input_data_1, path=input_path_1)

        input_data_2 = tp.event_set(
            timestamps=[2, 3, 8],
            features={"b": [1, 2, 3]},
        )
        tp.to_csv(input_data_2, path=input_path_2)

        # Create a graph.
        #
        # TODO: Update example with a multiple IO op when at least one
        # multi-IO op is implemented.
        input_node_1 = tp.input_node([("a", tp.float32)])
        input_node_2 = tp.input_node([("b", tp.float32)])
        output_node_1 = tp.moving_sum(input_node_1, 4)
        output_node_2 = tp.moving_sum(input_node_2, 4)

        # Execute the graph in Beam and export the result in a csv file.
        with TestPipeline() as p:
            input_beam_1 = p | tpb.read_csv(input_path_1, input_node_1.schema)
            input_beam_2 = p | tpb.read_csv(input_path_2, input_node_2.schema)

            outputs = tpb.run_multi_io(
                inputs={
                    input_node_1: input_beam_1,
                    input_node_2: input_beam_2,
                },
                outputs=[output_node_1, output_node_2],
            )
            outputs[output_node_1] | tpb.write_csv(
                output_path_1, output_node_1.schema, shard_name_template=""
            )
            outputs[output_node_2] | tpb.write_csv(
                output_path_2, output_node_2.schema, shard_name_template=""
            )
            p.run()

        with open(output_path_1, "r", encoding="utf-8") as f:
            print("Output 1:\n" + f.read(), flush=True)

        with open(output_path_2, "r", encoding="utf-8") as f:
            print("Output 2:\n" + f.read(), flush=True)


if __name__ == "__main__":
    absltest.main()
