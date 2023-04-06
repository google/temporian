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

from temporian.core import serialize
from temporian.core import processor
from temporian.core.test import utils


class SerializeTest(absltest.TestCase):
    def test_serialize(self):
        i1 = utils.create_input_event()
        o2 = utils.OpI1O1(i1)
        i3 = utils.create_input_event()
        o4 = utils.OpI2O1(o2.outputs["output"], i3)
        o5 = utils.OpI1O2(o4.outputs["output"])

        original = processor.infer_processor(
            {
                "io_input_1": i1,
                "io_input_2": i3,
            },
            {
                "io_output_1": o5.outputs["output_1"],
                "io_output_2": o4.outputs["output"],
            },
        )
        logging.info("original:\n%s", original)

        proto = serialize.serialize(original)
        logging.info("proto:\n%s", proto)

        restored = serialize.unserialize(proto)
        logging.info("restored:\n%s", restored)

        self.assertEqual(len(original.samplings), len(restored.samplings))
        self.assertEqual(len(original.features), len(restored.features))
        self.assertEqual(len(original.operators), len(restored.operators))
        self.assertEqual(len(original.events), len(restored.events))
        self.assertEqual(original.inputs.keys(), restored.inputs.keys())
        self.assertEqual(original.outputs.keys(), restored.outputs.keys())
        # TODO: Deep equality tests.

        # Ensures that "original" and "restored" don't link to the same objects.
        self.assertFalse(
            serialize.all_identifier(original.samplings)
            & serialize.all_identifier(restored.samplings)
        )
        self.assertFalse(
            serialize.all_identifier(original.features)
            & serialize.all_identifier(restored.features)
        )
        self.assertFalse(
            serialize.all_identifier(original.operators)
            & serialize.all_identifier(restored.operators)
        )
        self.assertFalse(
            serialize.all_identifier(original.events)
            & serialize.all_identifier(restored.events)
        )
        self.assertFalse(
            serialize.all_identifier(original.inputs.values())
            & serialize.all_identifier(restored.inputs.values())
        )
        self.assertFalse(
            serialize.all_identifier(original.outputs.values())
            & serialize.all_identifier(restored.outputs.values())
        )


if __name__ == "__main__":
    absltest.main()
