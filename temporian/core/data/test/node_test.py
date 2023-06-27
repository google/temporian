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

import temporian as tp
from temporian.core.test import utils
from temporian.implementation.numpy.data.event_set import EventSet


class NodeTest(absltest.TestCase):
    def test_evaluate_input(self):
        node = utils.create_source_node()
        evset = utils.create_input_event_set()
        result = node.run({node: evset})
        self.assertIsInstance(result, EventSet)
        self.assertTrue(result is evset)

    def test_evaluate_single_operator(self):
        evset = utils.create_input_event_set()
        result = tp.simple_moving_average(evset.node(), 10)
        result = result.run(evset)
        self.assertIsInstance(result, EventSet)


if __name__ == "__main__":
    absltest.main()
