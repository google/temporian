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
import os
import tempfile
import math

import temporian as tp


class TFPTest(absltest.TestCase):
    def test_evaluation(self):
        evset_1 = tp.event_set(
            timestamps=[0.0, 2.0, 4.0, 6.0],
            features={
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [5.0, 6.0, 7.0, 8.0],
            },
        )

        evset_2 = tp.event_set(timestamps=[1.0, 2.0, 2.0])

        i1 = evset_1.source_node()
        i2 = evset_2.source_node()

        h1 = tp.simple_moving_average(input=i1, window_length=7)
        h2 = tp.resample(input=h1, sampling=i2)
        h3 = i1 * 2.0 + 3.0 > 10.0

        result = tp.glue(tp.prefix("sma_", h2["f2"]), i2)

        result2 = tp.glue(tp.prefix("toto.", h3))

        result_data, result2_data = tp.evaluate(
            query=[result, result2],
            input={i1: evset_1, i2: evset_2},
            verbose=2,
        )
        logging.info("results: %s", result_data)
        logging.info("result2: %s", result2_data)

        with tempfile.TemporaryDirectory() as tempdir:
            result_data.plot(return_fig=True).savefig(
                os.path.join(tempdir, "p1.png")
            )
            tp.plot([evset_1, evset_2], return_fig=True).savefig(
                os.path.join(tempdir, "p2.png")
            )

    def test_serialization(self):
        a = tp.source_node([("f1", tp.float32), ("f2", tp.float32)])
        b = tp.simple_moving_average(input=a, window_length=7)

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_graph.tem")
            tp.save(inputs={"a": a}, outputs={"b": b}, path=path)

            inputs, outputs = tp.load(path=path)

        self.assertSetEqual(set(inputs.keys()), {"a"})
        self.assertSetEqual(set(outputs.keys()), {"b"})

    def test_serialization_single_node(self):
        a = tp.source_node(
            [("f1", tp.float32), ("f2", tp.float32)], name="my_source_node"
        )
        b = tp.simple_moving_average(input=a, window_length=7)
        b.name = "my_output_node"

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_graph.tem")
            tp.save(
                inputs=a,
                outputs=b,
                path=path,
            )

            inputs, outputs = tp.load(path=path)

        self.assertSetEqual(set(inputs.keys()), {"my_source_node"})
        self.assertSetEqual(set(outputs.keys()), {"my_output_node"})

    def test_serialization_squeeze_loading_results(self):
        a = tp.source_node(
            [("f1", tp.float32), ("f2", tp.float32)],
            name="my_source_node",
        )
        b = tp.simple_moving_average(input=a, window_length=7)
        b.name = "my_output_node"

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_graph.tem")
            tp.save(
                inputs=a,
                outputs=b,
                path=path,
            )

            i, o = tp.load(path=path, squeeze=True)

        self.assertEqual(i.name, "my_source_node")
        self.assertEqual(o.name, "my_output_node")

    def test_serialization_infer_inputs(self):
        a = tp.source_node(
            [("f1", tp.float32), ("f2", tp.float32)],
            name="my_source_node",
        )
        b = tp.simple_moving_average(input=a, window_length=7)
        b.name = "my_output_node"

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_graph.tem")
            tp.save(inputs=None, outputs=b, path=path)

            i, o = tp.load(path=path, squeeze=True)

        self.assertEqual(i.name, "my_source_node")
        self.assertEqual(o.name, "my_output_node")

    def test_list_registered_operators(self):
        logging.info("The operators:")
        for k, v in tp.get_operators().items():
            logging.info("  %s: %s", k, v)

    def test_event_set(self):
        evset = tp.event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "feature_1": [0.5, 0.6, math.nan, 0.9],
                "feature_2": ["red", "blue", "red", "blue"],
                "feature_3": [10, -1, 5, 5],
            },
            index_features=["feature_2"],
        )
        print(evset)


if __name__ == "__main__":
    absltest.main()
