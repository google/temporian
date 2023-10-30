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

    def test_add_index(self):
        self.assertTrue(isinstance(self.evset.add_index("a"), EventSet))
        self.assertTrue(isinstance(self.node.add_index("a"), EventSetNode))

    def test_begin(self):
        self.assertTrue(isinstance(self.evset.begin(), EventSet))
        self.assertTrue(isinstance(self.node.begin(), EventSetNode))

    def test_cast(self):
        self.assertTrue(isinstance(self.evset.cast({"a": float}), EventSet))
        self.assertTrue(isinstance(self.node.cast({"a": float}), EventSetNode))

    def test_cumsum(self):
        self.assertTrue(isinstance(self.evset.cumsum(), EventSet))
        self.assertTrue(isinstance(self.node.cumsum(), EventSetNode))

    def test_drop_index(self):
        self.assertTrue(isinstance(self.evset.drop_index("x"), EventSet))
        self.assertTrue(isinstance(self.node.drop_index("x"), EventSetNode))

    def test_end(self):
        self.assertTrue(isinstance(self.evset.end(), EventSet))
        self.assertTrue(isinstance(self.node.end(), EventSetNode))

    def test_enumerate(self):
        self.assertTrue(isinstance(self.evset.enumerate(), EventSet))
        self.assertTrue(isinstance(self.node.enumerate(), EventSetNode))

    def test_fast_fourier_transform(self):
        self.assertTrue(
            isinstance(
                self.evset["a"].experimental_fast_fourier_transform(
                    num_events=2
                ),
                EventSet,
            )
        )
        self.assertTrue(
            isinstance(
                self.node["a"].experimental_fast_fourier_transform(
                    num_events=2
                ),
                EventSetNode,
            )
        )

    def test_join(self):
        self.assertTrue(isinstance(self.evset.join(self.other_evset), EventSet))
        self.assertTrue(
            isinstance(self.node.join(self.other_node), EventSetNode)
        )

    def test_select(self):
        self.assertTrue(isinstance(self.evset.select("a"), EventSet))
        self.assertTrue(isinstance(self.node.select("a"), EventSetNode))

    def test_set_index(self):
        self.assertTrue(isinstance(self.evset.set_index("a"), EventSet))
        self.assertTrue(isinstance(self.node.set_index("a"), EventSetNode))

    def test_since_last(self):
        self.assertTrue(isinstance(self.evset.since_last(), EventSet))
        self.assertTrue(isinstance(self.node.since_last(), EventSetNode))

    def test_tick(self):
        self.assertTrue(isinstance(self.evset.tick(1), EventSet))
        self.assertTrue(isinstance(self.node.tick(1), EventSetNode))

    def test_timestamps(self):
        self.assertTrue(isinstance(self.evset.timestamps(), EventSet))
        self.assertTrue(isinstance(self.node.timestamps(), EventSetNode))

    def test_unique_timestamps(self):
        self.assertTrue(isinstance(self.evset.unique_timestamps(), EventSet))
        self.assertTrue(isinstance(self.node.unique_timestamps(), EventSetNode))

    def test_filter_moving_count(self):
        self.assertTrue(isinstance(self.evset.filter_moving_count(5), EventSet))
        self.assertTrue(
            isinstance(self.node.filter_moving_count(5), EventSetNode)
        )


if __name__ == "__main__":
    absltest.main()
