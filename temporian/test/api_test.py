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

from temporian.implementation.pandas.data.event import PandasEvent


class TFPTest(absltest.TestCase):
    def disabled_test_evaluation(self):
        a = t.input_event(
            [
                t.Feature(name="f1", dtype=t.dtype.FLOAT64),
                t.Feature(name="f2", dtype=t.dtype.FLOAT64),
            ]
        )

        b = t.sma(event=a, window_length=7)

        input_signal_data = PandasEvent(
            {
                "time": [0, 2, 4, 6],
                "f1": [1, 2, 3, 4],
                "f2": [5, 6, 7, 8],
            }
        )

        results = t.evaluate(
            query={"b": b},
            input_data={
                a: input_signal_data,
            },
        )
        logging.info("results: %s", results)

    def test_serialization(self):
        a = t.input_event(
            [
                t.Feature(name="f1"),
                t.Feature(name="f2"),
            ]
        )
        b = t.sma(event=a, window_length=7)

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
        b = t.sma(event=a, window_length=7)
        b.set_name("my_output_event")

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
        b = t.sma(event=a, window_length=7)
        b.set_name("my_output_event")

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_processor.tem")
            t.save(
                inputs=a,
                outputs=b,
                path=path,
            )

            i, o = t.load(path=path, squeeze=True)

        self.assertEqual(i.name(), "my_input_event")
        self.assertEqual(o.name(), "my_output_event")

    def test_serialization_infer_inputs(self):
        a = t.input_event(
            [
                t.Feature(name="f1"),
                t.Feature(name="f2"),
            ],
            name="my_input_event",
        )
        b = t.sma(event=a, window_length=7)
        b.set_name("my_output_event")

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_processor.tem")
            t.save(
                inputs=None,
                outputs=b,
                path=path
            )

            i, o = t.load(path=path, squeeze=True)

        self.assertEqual(i.name(), "my_input_event")
        self.assertEqual(o.name(), "my_output_event")


if __name__ == "__main__":
    absltest.main()
