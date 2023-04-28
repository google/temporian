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

import temporian as t
import pandas as pd
from temporian.implementation.numpy.data.event_set import EventSet


class TFPTest(absltest.TestCase):
    def test_evaluation(self):
        evset_1 = EventSet.from_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [0.0, 2.0, 4.0, 6.0],
                    "f1": [1.0, 2.0, 3.0, 4.0],
                    "f2": [5.0, 6.0, 7.0, 8.0],
                }
            )
        )

        evset_2 = EventSet.from_dataframe(
            pd.DataFrame({"timestamp": [1.0, 2.0, 2.0]})
        )

        i1 = evset_1.node()
        i2 = evset_2.node()
        h1 = t.simple_moving_average(node=i1, window_length=7.0)
        h2 = t.sample(node=h1, sampling=i2)
        result = t.glue(t.prefix("sma_", h2["f2"]), i2)

        result_data = t.evaluate(
            query=result,
            input_data={
                i1: evset_1,
                i2: evset_2,
            },
            verbose=2,
        )
        logging.info("results: %s", result_data)

        with tempfile.TemporaryDirectory() as tempdir:
            result_data.plot().savefig(os.path.join(tempdir, "p1.png"))
            t.plot([evset_1, evset_2]).savefig(os.path.join(tempdir, "p2.png"))

    def test_serialization(self):
        a = t.input_node(
            [
                t.Feature(name="f1"),
                t.Feature(name="f2"),
            ]
        )
        b = t.simple_moving_average(node=a, window_length=7.0)

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_processor.tem")
            t.save(
                inputs={"a": a},
                outputs={"b": b},
                path=path,
            )

            inputs, outputs = t.load(path=path)

        self.assertSetEqual(set(inputs.keys()), {"a"})
        self.assertSetEqual(set(outputs.keys()), {"b"})

    def test_serialization_single_event(self):
        a = t.input_event(
            [
                t.Feature(name="f1"),
                t.Feature(name="f2"),
            ],
            name="my_input_event",
        )
        b = t.simple_moving_average(event=a, window_length=7.0)
        b.name = "my_output_event"

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_processor.tem")
            t.save(
                inputs=a,
                outputs=b,
                path=path,
            )

            inputs, outputs = t.load(path=path)

        self.assertSetEqual(set(inputs.keys()), {"my_input_event"})
        self.assertSetEqual(set(outputs.keys()), {"my_output_event"})

    def test_serialization_squeeze_loading_results(self):
        a = t.input_event(
            [
                t.Feature(name="f1"),
                t.Feature(name="f2"),
            ],
            name="my_input_event",
        )
        b = t.simple_moving_average(event=a, window_length=7.0)
        b.name = "my_output_event"

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_processor.tem")
            t.save(
                inputs=a,
                outputs=b,
                path=path,
            )

            i, o = t.load(path=path, squeeze=True)

        self.assertEqual(i.name, "my_input_event")
        self.assertEqual(o.name, "my_output_event")

    def test_serialization_infer_inputs(self):
        a = t.input_event(
            [
                t.Feature(name="f1"),
                t.Feature(name="f2"),
            ],
            name="my_input_event",
        )
        b = t.simple_moving_average(event=a, window_length=7.0)
        b.name = "my_output_event"

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_processor.tem")
            t.save(inputs=None, outputs=b, path=path)

            i, o = t.load(path=path, squeeze=True)

        self.assertEqual(i.name, "my_input_event")
        self.assertEqual(o.name, "my_output_event")

    def test_list_registered_operators(self):
        logging.info("The operators:")
        for k, v in t.get_operators().items():
            logging.info("  %s: %s", k, v)


if __name__ == "__main__":
    absltest.main()
