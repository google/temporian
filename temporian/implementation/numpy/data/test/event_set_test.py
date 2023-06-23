import numpy as np
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set, IndexData

# TODO: Rename file to "event_set_test" and rename the following class to EventSetTest.


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
            indexes=["x", "y"],
        )

    def test_getitem(self):
        index_data = self._evset[(2, "world")]
        self.assertTrue(isinstance(index_data, IndexData))
        self.assertTrue(
            (np.array(index_data.features) == [[7, 8], [9, 10]]).all()
        )
        self.assertTrue((abs(index_data.timestamps - [0.4, 0.5]) < 1e-6).all())

    def test_getitem_error(self):
        with self.assertRaisesRegex(
            TypeError, "can only be accessed by index keys"
        ):
            self._evset["feature"]
        with self.assertRaisesRegex(
            TypeError, "can only be accessed by index keys"
        ):
            self._evset[0]

    def test_setitem_error(self):
        with self.assertRaisesRegex(
            TypeError, "not intended to be modified externally"
        ):
            self._evset["feature"] = None
        with self.assertRaisesRegex(
            TypeError, "not intended to be modified externally"
        ):
            self._evset[(2, "world")] = None

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
memory usage: 1.2 kB
""",
        )

    def test_memory_usage(self):
        memory_usage = self._evset.memory_usage()
        print("memory_usage:", memory_usage)

        self.assertLessEqual(memory_usage, 1200 + 500)
        self.assertGreaterEqual(memory_usage, 1200 - 500)


if __name__ == "__main__":
    absltest.main()
