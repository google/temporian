from absl.testing import absltest
from temporian.core.data.node import EventSetNode
from temporian.implementation.numpy.data.event_set import EventSet

from temporian.implementation.numpy.data.io import event_set


class EventSetOpsTest(absltest.TestCase):
    def setUp(self):
        self.evset = event_set(
            timestamps=[0.1, 0.2, 0.3, 0.4, 0.5],
            features={
                "a": [1, 2, 3, 7, 8],
                "b": [4, 5, 6, 9, 10],
                "x": [1, 1, 1, 2, 2],
                "y": ["hello", "hello", "hello", "world", "world"],
            },
            indexes=["x", "y"],
        )
        self.node = self.evset.node()

    def test_add_index(self):
        self.assertTrue(isinstance(self.evset.add_index("a"), EventSet))
        self.assertTrue(isinstance(self.node.add_index("a"), EventSetNode))

    def test_begin(self):
        self.assertTrue(isinstance(self.evset.begin(), EventSet))
        self.assertTrue(isinstance(self.node.begin(), EventSetNode))

    def test_cast(self):
        self.assertTrue(isinstance(self.evset.cast({"a": float}), EventSet))
        self.assertTrue(isinstance(self.node.cast({"a": float}), EventSetNode))

    def test_end(self):
        self.assertTrue(isinstance(self.evset.end(), EventSet))
        self.assertTrue(isinstance(self.node.end(), EventSetNode))

    def test_enumerate(self):
        self.assertTrue(isinstance(self.evset.enumerate(), EventSet))
        self.assertTrue(isinstance(self.node.enumerate(), EventSetNode))

    def test_filter(self):
        self.assertTrue(
            isinstance(self.evset.filter(self.evset["a"] > 3), EventSet)
        )
        self.assertTrue(
            isinstance(self.node.filter(self.node["a"] > 3), EventSetNode)
        )

    def test_drop_index(self):
        self.assertTrue(isinstance(self.evset.drop_index("x"), EventSet))
        self.assertTrue(isinstance(self.node.drop_index("x"), EventSetNode))

    def test_set_index(self):
        self.assertTrue(isinstance(self.evset.set_index("a"), EventSet))
        self.assertTrue(isinstance(self.node.set_index("a"), EventSetNode))


if __name__ == "__main__":
    absltest.main()
