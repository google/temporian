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
import re

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

        i1 = evset_1.node()
        i2 = evset_2.node()

        h1 = i1.simple_moving_average(window_length=7)
        h2 = h1.resample(i2)
        h3 = i1 * 2.0 + 3.0 > 10.0

        result = tp.glue(h2["f2"].prefix("sma_"), i2)

        result2 = tp.glue(h3.prefix("toto."))

        result_data, result2_data = tp.run(
            query=[result, result2],
            input={i1: evset_1, i2: evset_2},
            verbose=2,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            result_data.plot(return_fig=True).savefig(
                os.path.join(tempdir, "p1.png")
            )
            tp.plot([evset_1, evset_2], return_fig=True).savefig(
                os.path.join(tempdir, "p2.png")
            )

    def test_eager_mode(self):
        evset_1 = tp.event_set(
            timestamps=[0.0, 2.0, 4.0, 6.0],
            features={
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [5.0, 6.0, 7.0, 8.0],
            },
        )

        evset_2 = tp.event_set(timestamps=[1.0, 2.0, 2.0])

        h1 = evset_1.simple_moving_average(window_length=7)
        h2 = h1.resample(evset_2)

        self.assertTrue(isinstance(h1, tp.EventSet))
        self.assertTrue(isinstance(h2, tp.EventSet))

        del h1

        h3 = evset_1 * 2.0 + 3.0 > 10.0

        result = tp.glue(h2["f2"].prefix("sma_"), evset_2)

        result2 = tp.glue(h3.prefix("toto."))

        self.assertTrue(isinstance(result, tp.EventSet))
        self.assertTrue(isinstance(result2, tp.EventSet))

    def test_eager_mixed_args(self):
        evset = tp.event_set(timestamps=[0.0])

        with self.assertRaises(ValueError):
            evset.simple_moving_average(window_length=7, sampling=evset.node())

        with self.assertRaises(ValueError):
            evset.node().simple_moving_average(window_length=7, sampling=evset)

    def test_pandas(self):
        evset = tp.event_set(
            timestamps=[0.0, 2.0, 4.0, 6.0],
            features={
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [5.0, 6.0, 7.0, 8.0],
            },
        )

        df = tp.to_pandas(evset)
        reconstructed_evset = tp.from_pandas(df)
        self.assertEqual(evset, reconstructed_evset)

    def test_polars(self):
        evset = tp.event_set(
            timestamps=[0.0, 2.0, 4.0, 6.0],
            features={
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [5.0, 6.0, 7.0, 8.0],
            },
        )

        df = tp.to_polars(evset)
        reconstructed_evset = tp.from_polars(df)
        self.assertEqual(evset, reconstructed_evset)

    def test_serialization(self):
        a = tp.input_node([("f1", tp.float32), ("f2", tp.float32)])
        b = a.simple_moving_average(window_length=7)

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_graph.tem")
            tp.save_graph(inputs={"a": a}, outputs={"b": b}, path=path)

            inputs, outputs = tp.load_graph(path=path)

        self.assertSetEqual(set(inputs.keys()), {"a"})
        self.assertSetEqual(set(outputs.keys()), {"b"})

    def test_serialization_single_node(self):
        a = tp.input_node(
            [("f1", tp.float32), ("f2", tp.float32)], name="my_source_node"
        )
        b = a.simple_moving_average(window_length=7)
        b.name = "my_output_node"

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_graph.tem")
            tp.save_graph(
                inputs=a,
                outputs=b,
                path=path,
            )

            inputs, outputs = tp.load_graph(path=path)

        self.assertSetEqual(set(inputs.keys()), {"my_source_node"})
        self.assertSetEqual(set(outputs.keys()), {"my_output_node"})

    def test_serialization_squeeze_loading_results(self):
        a = tp.input_node(
            [("f1", tp.float32), ("f2", tp.float32)],
            name="my_source_node",
        )
        b = a.simple_moving_average(window_length=7)
        b.name = "my_output_node"

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_graph.tem")
            tp.save_graph(
                inputs=a,
                outputs=b,
                path=path,
            )

            i, o = tp.load_graph(path=path, squeeze=True)

        self.assertEqual(i.name, "my_source_node")
        self.assertEqual(o.name, "my_output_node")

    def test_serialization_infer_inputs(self):
        a = tp.input_node(
            [("f1", tp.float32), ("f2", tp.float32)],
            name="my_source_node",
        )
        b = a.simple_moving_average(window_length=7)
        b.name = "my_output_node"

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_graph.tem")
            tp.save_graph(inputs=None, outputs=b, path=path)

            i, o = tp.load_graph(path=path, squeeze=True)

        self.assertEqual(i.name, "my_source_node")
        self.assertEqual(o.name, "my_output_node")

    def test_event_set(self):
        tp.event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "feature_1": [0.5, 0.6, math.nan, 0.9],
                "feature_2": ["red", "blue", "red", "blue"],
                "feature_3": [10, -1, 5, 5],
            },
            indexes=["feature_2"],
        )

    def test_wrong_type(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Non matching type for \"<class 'list'>\" in the union"),
        ):
            tp.event_set(timestamps=[1, 2], features=[])

    def test_duration_to_string(self):
        self.assertEqual(
            tp.duration.to_string(tp.duration.days(2) + tp.duration.hours(3)),
            "2d3h",
        )

    def test_schema_to_from_proto(self):
        a = tp.Schema(features=[("f1", tp.int32), ("f2", tp.float64)])
        p = a.to_proto()
        b = tp.Schema.from_proto(p)
        self.assertEqual(a, b)

    def test_schema_to_from_proto_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "schema.pbtxt")
            a = tp.Schema(features=[("f1", tp.int32), ("f2", tp.float64)])
            a.to_proto_file(path)
            b = tp.Schema.from_proto_file(path)
            self.assertEqual(a, b)


if __name__ == "__main__":
    absltest.main()
