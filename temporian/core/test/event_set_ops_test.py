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
from temporian.core.data.node import EventSetNode
from temporian.implementation.numpy.data.event_set import EventSet

from temporian.implementation.numpy.data.io import event_set


# TODO: remove all the operator tests from this class when done migrating them
# Leave a single test, checking that the output of an op is an EventSet when
# passed an EventSet and an EventSetNode when passed an EventSetNode
class EventSetOpsTest(absltest.TestCase):
    """Tests that all expected operators are available and work on EventSet and
    EventSetNode and that they return the correct type."""

    def setUp(self):
        self.evset = event_set(
            timestamps=[0.1, 0.2, 0.3, 0.4, 0.5],
            features={
                "a": [1.0, 2.0, 3.0, 7.0, 8.0],
                "b": [4.0, 5.0, 6.0, 9.0, 10.0],
                "x": [1, 1, 1, 2, 2],
                "y": ["hello", "hello", "hello", "world", "world"],
            },
            indexes=["x", "y"],
            is_unix_timestamp=True,
        )
        self.other_evset = event_set(
            timestamps=[0.4, 0.5, 0.6, 0.7],
            features={
                "c": [11, 12, 13, 14],
                "x": [1, 1, 1, 2],
                "y": ["hello", "hello", "hello", "world"],
            },
            indexes=["x", "y"],
            is_unix_timestamp=True,
        )
        self.node = self.evset.node()
        self.other_node = self.other_evset.node()

    def test_cast(self):
        self.assertTrue(isinstance(self.evset.cast({"a": float}), EventSet))
        self.assertTrue(isinstance(self.node.cast({"a": float}), EventSetNode))

    def test_cumsum(self):
        self.assertTrue(isinstance(self.evset.cumsum(), EventSet))
        self.assertTrue(isinstance(self.node.cumsum(), EventSetNode))

    def test_enumerate(self):
        self.assertTrue(isinstance(self.evset.enumerate(), EventSet))
        self.assertTrue(isinstance(self.node.enumerate(), EventSetNode))

    def test_select(self):
        self.assertTrue(isinstance(self.evset.select("a"), EventSet))
        self.assertTrue(isinstance(self.node.select("a"), EventSetNode))

    def test_timestamps(self):
        self.assertTrue(isinstance(self.evset.timestamps(), EventSet))
        self.assertTrue(isinstance(self.node.timestamps(), EventSetNode))

    def test_filter_moving_count(self):
        self.assertTrue(isinstance(self.evset.filter_moving_count(5), EventSet))
        self.assertTrue(
            isinstance(self.node.filter_moving_count(5), EventSetNode)
        )


if __name__ == "__main__":
    absltest.main()
