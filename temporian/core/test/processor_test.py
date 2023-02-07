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

from temporian.core import processor
from temporian.core.data import dtype
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base
from temporian.proto import core_pb2 as pb
from temporian.core.test import utils


class ProcessorTest(absltest.TestCase):
    def test_dependency_basic(self):
        i1 = utils.create_input_event()
        o2 = utils.OpI1O1(i1)
        i3 = utils.create_input_event()
        o4 = utils.OpI2O1(o2.outputs()["output"], i3)
        o5 = utils.OpI1O2(o4.outputs()["output"])

        p = processor.infer_processor(
            {
                "io_input_1": i1,
                "io_input_2": i3,
            },
            {
                "io_output_1": o5.outputs()["output_1"],
                "io_output_2": o4.outputs()["output"],
            },
        )
        logging.info("Processor:\n%s", p)

        # The two input operators are not listed.
        self.assertLen(p.operators(), 3)

        # The two samplings created by the two input operators.
        self.assertLen(p.samplings(), 2)
        self.assertLen(p.events(), 6)
        self.assertLen(p.features(), 8)

    def test_dependency_missing_input(self):
        i = utils.create_input_event()
        o2 = utils.OpI1O1(i)

        with self.assertRaisesRegex(
            ValueError,
            "Missing input features.",
        ):
            processor.infer_processor({}, {"io_output": o2.outputs()["output"]})

    def test_automatic_input(self):
        i1 = utils.create_input_event()
        i1.set_name("io_input_1")
        o2 = utils.OpI1O1(i1)
        i3 = utils.create_input_event()
        i3.set_name("io_input_2")
        o4 = utils.OpI2O1(o2.outputs()["output"], i3)
        o5 = utils.OpI1O2(o4.outputs()["output"])

        p = processor.infer_processor(
            None,
            {
                "io_output_1": o5.outputs()["output_1"],
                "io_output_2": o4.outputs()["output"],
            },
            infer_inputs=True,
        )
        logging.info("Processor:\n%s", p)

        # The two input operators are not listed.
        self.assertLen(p.operators(), 3)

        # The two samplings created by the two input operators.
        self.assertLen(p.samplings(), 2)
        self.assertLen(p.events(), 6)
        self.assertLen(p.features(), 8)

    def test_automatic_input_missing_name(self):
        i1 = utils.create_input_event()
        o2 = utils.OpI1O1(i1)

        with self.assertRaisesRegex(ValueError, "Infered input without a name"):
            processor.infer_processor(
                None,
                {"io_output_1": o2.outputs()["output"]},
                infer_inputs=True,
            )

    def test_automatic_input_missing_name(self):
        i1 = utils.create_input_event()
        o2 = utils.OpI1O1(i1)

        with self.assertRaisesRegex(ValueError, "Infered input without a name"):
            processor.infer_processor(
                None,
                {"io_output_1": o2.outputs()["output"]},
                infer_inputs=True,
            )


if __name__ == "__main__":
    absltest.main()
