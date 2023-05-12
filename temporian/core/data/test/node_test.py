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
        node = utils.create_input_node()
        evset = utils.create_input_event_set()

        result = node.evaluate({node: evset})

        self.assertIsInstance(result, EventSet)
        self.assertTrue(result is evset)

    def test_evaluate_single_operator(self):
        a = utils.create_input_node(name="node")
        evset = utils.create_input_event_set()

        sma = tp.simple_moving_average(a, 10)

        result = sma.evaluate({"node": evset})

        self.assertIsInstance(result, EventSet)


if __name__ == "__main__":
    absltest.main()
