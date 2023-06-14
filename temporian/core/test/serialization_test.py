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
import temporian as tp
from temporian.core.serialization import serialize
from temporian.core import graph
from temporian.core.data.dtypes.dtype import DType
from temporian.core.test import utils
from temporian.implementation.numpy.data.io import event_set


class SerializeTest(absltest.TestCase):
    def test_serialize(self):
        i1 = utils.create_source_node()
        o2 = utils.OpI1O1(i1)
        i3 = utils.create_source_node()
        o4 = utils.OpI2O1(o2.outputs["output"], i3)
        o5 = utils.OpI1O2(o4.outputs["output"])

        original = graph.infer_graph_named_nodes(
            {"io_input_1": i1, "io_input_2": i3},
            {
                "io_output_1": o5.outputs["output_1"],
                "io_output_2": o4.outputs["output"],
            },
        )
        logging.info("original:\n%s", original)

        proto = serialize._serialize(original)
        logging.info("proto:\n%s", proto)

        restored = serialize._unserialize(proto)
        logging.info("restored:\n%s", restored)

        self.assertEqual(len(original.samplings), len(restored.samplings))
        self.assertEqual(len(original.features), len(restored.features))
        self.assertEqual(len(original.operators), len(restored.operators))
        self.assertEqual(len(original.nodes), len(restored.nodes))
        self.assertEqual(
            original.named_inputs.keys(), restored.named_inputs.keys()
        )
        self.assertEqual(
            original.named_outputs.keys(), restored.named_outputs.keys()
        )
        # TODO: Deep equality tests.

        # Ensures that "original" and "restored" don't link to the same objects.
        self.assertFalse(
            serialize._all_identifiers(original.samplings)
            & serialize._all_identifiers(restored.samplings)
        )
        self.assertFalse(
            serialize._all_identifiers(original.features)
            & serialize._all_identifiers(restored.features)
        )
        self.assertFalse(
            serialize._all_identifiers(original.operators)
            & serialize._all_identifiers(restored.operators)
        )
        self.assertFalse(
            serialize._all_identifiers(original.nodes)
            & serialize._all_identifiers(restored.nodes)
        )
        self.assertFalse(
            serialize._all_identifiers(original.named_inputs.values())
            & serialize._all_identifiers(restored.named_inputs.values())
        )
        self.assertFalse(
            serialize._all_identifiers(original.named_outputs.values())
            & serialize._all_identifiers(restored.named_outputs.values())
        )

    def test_serialize_autonode(self):
        input_data = event_set(
            timestamps=[1, 2, 3, 4],
            features={"f1": [5, 6, 7, 8], "x": [1, 1, 2, 2]},
            index_features=["x"],
        )

        input_node = input_data.node()
        output_node = tp.simple_moving_average(input_node, 2.0)

        original = graph.infer_graph_named_nodes(
            {"i": input_node},
            {"o": output_node},
        )
        logging.info("original:\n%s", original)

        proto = serialize._serialize(original)
        logging.info("proto:\n%s", proto)

        restored = serialize._unserialize(proto)
        logging.info("restored:\n%s", restored)

    def test_serialize_attributes(self):
        """Test serialization with different types of operator attributes."""
        attributes = {
            "attr_int": 1,
            "attr_str": "hello",
            "attr_list": ["temporian", "rocks"],
            "attr_float": 5.0,
            "attr_bool": True,
            "attr_map": {"good": "bye", "nice": "to", "meet": "you"},
            "attr_list_dtypes": [DType.FLOAT32, DType.STRING],
        }
        i_event = utils.create_source_node()
        operator = utils.OpWithAttributes(i_event, **attributes)

        original = graph.infer_graph_named_nodes(
            inputs={"i_event": i_event},
            outputs={"output": operator.outputs["output"]},
        )
        logging.info("original:\n%s", original)

        proto = serialize._serialize(original)
        logging.info("proto:\n%s", proto)

        restored = serialize._unserialize(proto)
        logging.info("restored:\n%s", restored)

        self.assertEqual(len(original.operators), len(restored.operators))
        self.assertEqual(
            original.named_inputs.keys(), restored.named_inputs.keys()
        )
        self.assertEqual(
            original.named_outputs.keys(), restored.named_outputs.keys()
        )

        # Check all restored attributes for the only operator
        restored_attributes = list(restored.operators)[0].attributes
        for attr_name, attr_value in attributes.items():
            self.assertEqual(attr_value, restored_attributes[attr_name])

        self.assertFalse(
            serialize._all_identifiers(original.operators)
            & serialize._all_identifiers(restored.operators)
        )
        self.assertFalse(
            serialize._all_identifiers(original.named_inputs.values())
            & serialize._all_identifiers(restored.named_inputs.values())
        )
        self.assertFalse(
            serialize._all_identifiers(original.named_outputs.values())
            & serialize._all_identifiers(restored.named_outputs.values())
        )

    def test_serialize_and_run(self):
        input_data = tp.event_set(
            timestamps=[1, 2, 3], features={"x": [1, 2, 3], "y": [4, 5, 6]}
        )

        input_node = input_data.node()
        x = input_node
        x = tp.cast(x, float)
        x = x["x"]
        x = 2 * x
        output_node = x

        result = tp.evaluate(output_node, {input_node: input_data})
        print("result:", result)

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "my_graph.tem")
            tp.save(
                inputs={"a": input_node}, outputs={"b": output_node}, path=path
            )
            loaded_inputs, loaded_outputs = tp.load(path=path, squeeze=True)

        loaded_results = tp.evaluate(
            loaded_outputs, {loaded_inputs: input_data}
        )
        print("loaded_results:", loaded_results)

        self.assertEqual(result, loaded_results)


if __name__ == "__main__":
    absltest.main()
