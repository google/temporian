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

from absl.testing import absltest

from temporian.core.test import utils


class NodeTest(absltest.TestCase):
    def test_hash_map(self):
        """
        These tests are the reason to not overwrite __eq__ in Node
        """
        node_list = []
        node_map = {}
        for i in range(100):
            node_name = f"node_{i}"
            node = utils.create_input_node(name=node_name)
            node_list.append(node)
            node_map[node] = node_name

        for idx, node in enumerate(node_list):
            assert idx == node_list.index(node)
            assert node_map[node] == node.name
            assert node in node_list
            assert node in node_map


if __name__ == "__main__":
    absltest.main()
