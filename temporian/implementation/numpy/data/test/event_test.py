from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set

# Load all the implementations
from temporian.implementation.numpy.operators import all_operators as _impls


class EventTest(absltest.TestCase):
    def setUp(self):
        self._evset = event_set(
            timestamps=[0.1, 0.2, 0.3, 0.4, 0.5],
            features={
                "a": [1, 2, 3, 7, 8],
                "b": [4, 5, 6, 9, 10],
                "x": [1, 1, 1, 2, 2],
                "y": ["hello", "hello", "hello", "world", "world"],
            },
            index_features=["x", "y"],
        )

    def test_data_access(self):
        self.assertEqual(
            repr(self._evset.schema.features), "[('a', int64), ('b', int64)]"
        )
        self.assertEqual(
            repr(self._evset.schema.indexes), "[('x', int64), ('y', str_)]"
        )

    def test_repr(self):
        print(self._evset)
        self.assertEqual(
            repr(self._evset),
            """indexes: [('x', int64), ('y', str_)]
features: [('a', int64), ('b', int64)]
events:
    x=1 y=hello (3 events):
        timestamps: [0.1 0.2 0.3]
        'a': [1 2 3]
        'b': [4 5 6]
    x=2 y=world (2 events):
        timestamps: [0.4 0.5]
        'a': [7 8]
        'b': [ 9 10]
""",
        )


if __name__ == "__main__":
    absltest.main()
